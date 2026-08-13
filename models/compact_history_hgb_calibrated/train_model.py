import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_FILENAME = "model.pkl"

CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "count_state",
    "hand_matchup",
]

RATE_SPECS = [
    ("asof_pitcher_n", "asof_pitcher_success_rate", "pitcher_success"),
    ("asof_pitcher_n", "asof_pitcher_reverse_rate", "pitcher_reverse"),
    ("asof_pitcher_n", "asof_pitcher_middle_rate", "pitcher_middle"),
    ("asof_pitcher_n", "asof_pitcher_ball_rate", "pitcher_ball"),
    ("asof_pitcher_n", "asof_pitcher_strike_rate", "pitcher_strike"),
    ("asof_batter_n", "asof_batter_success_rate", "batter_success"),
    ("asof_batter_n", "asof_batter_middle_rate", "batter_middle"),
    ("asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "pitcher_fastball"),
    ("asof_pitcher_pitchmix_n", "asof_pitcher_breaking_rate", "pitcher_breaking"),
    ("asof_pitcher_pitchmix_n", "asof_pitcher_offspeed_rate", "pitcher_offspeed"),
]

SMOOTHING_STRENGTHS = [50, 800]

BASE_FEATURES = [
    "season",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "li",
    "pitcher_hand",
    "batter_hand",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

DERIVED_FEATURES = [
    "count_state",
    "hand_matchup",
    "is_full_count",
    "is_pitcher_ahead",
    "is_hitter_ahead",
    "is_even_count",
    "has_runner",
    "has_risp",
    "bases_loaded",
    "pressure",
    "log_pitcher_n",
    "log_batter_n",
    "log_pitchmix_n",
    "pitcher_recent5_success_delta",
    "pitcher_recent3_success_delta",
    "pitcher_recent1_success_delta",
    "pitcher_recent5_middle_delta",
    "pitcher_ball_minus_strike_rate",
    "pitcher_reverse_plus_middle_rate",
    "batter_success_minus_pitcher_success",
    "batter_middle_minus_pitcher_middle",
    "fastball_minus_breaking_rate",
    "offspeed_share_of_nonfastball",
]

def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_train_path() -> Path:
    return find_repo_root() / "data" / "train.csv"


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "model" / MODEL_FILENAME


def load_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {ID_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"train data is missing required columns: {sorted(missing)}")
    return df


def make_rate_priors(train: pd.DataFrame) -> dict[str, float]:
    priors = {}
    target_mean = float(train[TARGET_COL].mean())
    for _, rate_col, _ in RATE_SPECS:
        if rate_col in train.columns:
            value = train[rate_col].mean(skipna=True)
            priors[rate_col] = float(value) if pd.notna(value) else target_mean
    priors[TARGET_COL] = target_mean
    return priors


def smooth_rate(df: pd.DataFrame, n_col: str, rate_col: str, prior: float, strength: int) -> pd.Series:
    n = pd.to_numeric(df[n_col], errors="coerce").fillna(0).clip(lower=0)
    rate = pd.to_numeric(df[rate_col], errors="coerce").fillna(prior)
    return (n * rate + strength * prior) / (n + strength)


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan).fillna(0)


def build_features(df: pd.DataFrame, rate_priors: dict[str, float]) -> pd.DataFrame:
    x = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").copy()

    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    x["hand_matchup"] = x["pitcher_hand"].astype(str) + "_" + x["batter_hand"].astype(str)

    x["is_full_count"] = ((x["balls_before"] == 3) & (x["strikes_before"] == 2)).astype("int8")
    x["is_pitcher_ahead"] = (x["strikes_before"] > x["balls_before"]).astype("int8")
    x["is_hitter_ahead"] = (x["balls_before"] > x["strikes_before"]).astype("int8")
    x["is_even_count"] = (x["balls_before"] == x["strikes_before"]).astype("int8")
    x["has_runner"] = (x["num_runners_on"] > 0).astype("int8")
    x["has_risp"] = ((x["runner_on_2b"] == 1) | (x["runner_on_3b"] == 1)).astype("int8")
    x["bases_loaded"] = (x["num_runners_on"] == 3).astype("int8")

    x["pressure"] = x["li"] * (1 + x["num_runners_on"])

    x["log_pitcher_n"] = np.log1p(x["asof_pitcher_n"].clip(lower=0))
    x["log_batter_n"] = np.log1p(x["asof_batter_n"].clip(lower=0))
    x["log_pitchmix_n"] = np.log1p(x["asof_pitcher_pitchmix_n"].clip(lower=0))

    for n_col, rate_col, prefix in RATE_SPECS:
        prior = rate_priors.get(rate_col, rate_priors.get(TARGET_COL, 0.5))
        for strength in SMOOTHING_STRENGTHS:
            x[f"{prefix}_smooth_{strength}"] = smooth_rate(x, n_col, rate_col, prior, strength)

    x["pitcher_recent5_success_delta"] = (
        x["asof_pitcher_prev5_game_success_rate"] - x["asof_pitcher_success_rate"]
    )
    x["pitcher_recent3_success_delta"] = (
        x["asof_pitcher_prev3_game_success_rate"] - x["asof_pitcher_success_rate"]
    )
    x["pitcher_recent1_success_delta"] = (
        x["asof_pitcher_prev1_game_success_rate"] - x["asof_pitcher_success_rate"]
    )
    x["pitcher_recent5_middle_delta"] = (
        x["asof_pitcher_prev5_game_middle_rate"] - x["asof_pitcher_middle_rate"]
    )
    x["pitcher_ball_minus_strike_rate"] = x["asof_pitcher_ball_rate"] - x["asof_pitcher_strike_rate"]
    x["pitcher_reverse_plus_middle_rate"] = (
        x["asof_pitcher_reverse_rate"] + x["asof_pitcher_middle_rate"]
    )
    x["batter_success_minus_pitcher_success"] = (
        x["asof_batter_success_rate"] - x["asof_pitcher_success_rate"]
    )
    x["batter_middle_minus_pitcher_middle"] = (
        x["asof_batter_middle_rate"] - x["asof_pitcher_middle_rate"]
    )
    x["fastball_minus_breaking_rate"] = (
        x["asof_pitcher_fastball_rate"] - x["asof_pitcher_breaking_rate"]
    )
    x["offspeed_share_of_nonfastball"] = safe_divide(
        x["asof_pitcher_offspeed_rate"],
        1 - x["asof_pitcher_fastball_rate"],
    )

    smooth_features = [
        f"{prefix}_smooth_{strength}"
        for _, _, prefix in RATE_SPECS
        for strength in SMOOTHING_STRENGTHS
    ]
    keep_cols = [col for col in BASE_FEATURES + DERIVED_FEATURES + smooth_features if col in x.columns]
    return x[keep_cols]


def make_pipeline(feature_columns: list[str]) -> Pipeline:
    cat_cols = [col for col in CAT_COLS if col in feature_columns]
    num_cols = [col for col in feature_columns if col not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ],
        remainder="drop",
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=120,
        l2_regularization=0.05,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )

    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def apply_calibration(pred: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    return sigmoid(calibration["scale"] * logit(pred) + calibration["bias"])


def strip_predict_unused_rng(model: Pipeline) -> None:
    clf = model.named_steps.get("clf")
    if clf is not None and hasattr(clf, "_feature_subsample_rng"):
        clf._feature_subsample_rng = None


def fit_logit_brier_calibration(pred: np.ndarray, y: pd.Series) -> dict[str, float]:
    z = logit(pred)
    y_values = y.to_numpy(dtype="float64")

    def objective(params: np.ndarray) -> float:
        scale, bias = params
        calibrated = sigmoid(scale * z + bias)
        return float(np.mean((calibrated - y_values) ** 2))

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0]),
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-10},
    )
    scale, bias = result.x
    return {
        "scale": float(scale),
        "bias": float(bias),
        "success": bool(result.success),
        "objective": float(result.fun),
    }


def score_dict(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    brier = brier_score_loss(y_true, pred)
    mean_control_rate = float(y_true.mean())
    mean_control_brier = mean_control_rate * (1 - mean_control_rate)
    return {
        "auc": float(roc_auc_score(y_true, pred)),
        "logloss": float(log_loss(y_true, pred)),
        "brier": float(brier),
        "bss_score": float(max(0, 100000 * (1 - brier / mean_control_brier))),
        "mean_control_rate": mean_control_rate,
        "mean_control_brier": float(mean_control_brier),
        "ap": float(average_precision_score(y_true, pred)),
        "mean_pred": float(pred.mean()),
    }


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name}: "
        f"auc={metrics['auc']:.6f} "
        f"logloss={metrics['logloss']:.6f} "
        f"brier={metrics['brier']:.6f} "
        f"bss_score={metrics['bss_score']:.6f} "
        f"mean_control_rate={metrics['mean_control_rate']:.6f} "
        f"mean_control_brier={metrics['mean_control_brier']:.6f} "
        f"ap={metrics['ap']:.6f} "
        f"mean_pred={metrics['mean_pred']:.6f}"
    )


def train_and_validate(train: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict]]:
    if train["season"].nunique() < 2 or 2024 not in set(train["season"]):
        print("Skip holdout validation: season 2024 is not available.")
        return {"scale": 1.0, "bias": 0.0, "success": True, "objective": np.nan}, {}

    fit_df = train[train["season"] <= 2023].copy()
    valid_df = train[train["season"] == 2024].copy()
    if fit_df.empty or valid_df.empty:
        print("Skip holdout validation: not enough rows.")
        return {"scale": 1.0, "bias": 0.0, "success": True, "objective": np.nan}, {}

    rate_priors = make_rate_priors(fit_df)
    x_fit = build_features(fit_df, rate_priors)
    x_valid = build_features(valid_df, rate_priors)
    model = make_pipeline(list(x_fit.columns))

    start = time.time()
    model.fit(x_fit, fit_df[TARGET_COL])
    elapsed = time.time() - start
    raw_pred = model.predict_proba(x_valid)[:, 1]
    calibration = fit_logit_brier_calibration(raw_pred, valid_df[TARGET_COL])
    calibrated_pred = apply_calibration(raw_pred, calibration)
    raw_metrics = score_dict(valid_df[TARGET_COL], raw_pred)
    calibrated_metrics = score_dict(valid_df[TARGET_COL], calibrated_pred)

    print(f"Validation fit rows={len(x_fit)} valid rows={len(x_valid)} seconds={elapsed:.1f}")
    print_metrics("holdout_2024_raw", raw_metrics)
    print_metrics("holdout_2024_calibrated", calibrated_metrics)
    print(
        "calibration: "
        f"scale={calibration['scale']:.8f} "
        f"bias={calibration['bias']:.8f} "
        f"success={calibration['success']}"
    )
    return calibration, {"raw": raw_metrics, "calibrated": calibrated_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=default_train_path())
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    print(f"Load train: {args.train_path}")
    train = load_train(args.train_path)
    rate_priors = make_rate_priors(train)
    print(f"Train rows={len(train)} target_mean={rate_priors[TARGET_COL]:.6f}")

    calibration = {"scale": 1.0, "bias": 0.0, "success": True, "objective": np.nan}
    validation_metrics = {}
    if not args.skip_validation:
        calibration, validation_metrics = train_and_validate(train)

    print("Build final features...")
    x_train = build_features(train, rate_priors)
    model = make_pipeline(list(x_train.columns))

    start = time.time()
    model.fit(x_train, train[TARGET_COL])
    strip_predict_unused_rng(model)
    elapsed = time.time() - start
    print(f"Final fit rows={len(x_train)} features={x_train.shape[1]} seconds={elapsed:.1f}")

    artifact = {
        "model": model,
        "feature_columns": list(x_train.columns),
        "cat_columns": [col for col in CAT_COLS if col in x_train.columns],
        "rate_priors": rate_priors,
        "calibration": calibration,
        "validation_metrics": validation_metrics,
        "target_col": TARGET_COL,
        "id_col": ID_COL,
        "sklearn_version": sklearn.__version__,
    }

    os.makedirs(args.model_path.parent, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
