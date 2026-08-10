import argparse
import os
import time
from dataclasses import dataclass
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_FILENAME = "model.pkl"

CAT_COLS = ["top_bottom", "game_type", "base_state"]

SEASON_WEIGHTS = {
    2019: 0.7,
    2020: 0.8,
    2021: 0.9,
    2022: 1.0,
    2023: 1.2,
    2024: 1.5,
}


@dataclass(frozen=True)
class TargetEncodingSpec:
    name: str
    columns: tuple[str, ...]
    smoothing: float


TE_SPECS = [
    TargetEncodingSpec("te_pitcher_id", ("pitcher_id",), 500),
    TargetEncodingSpec("te_pitcher_batter_hand", ("pitcher_id", "batter_hand"), 800),
    TargetEncodingSpec("te_pitcher_count", ("pitcher_id", "balls_before", "strikes_before"), 1200),
    TargetEncodingSpec("te_count", ("balls_before", "strikes_before"), 2000),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_train_path() -> Path:
    return repo_root() / "data" / "train.csv"


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "model" / MODEL_FILENAME


def load_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {ID_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"train data is missing required columns: {sorted(missing)}")
    return df


def make_group_key(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if len(columns) == 1:
        return df[columns[0]].astype("string").fillna("__NA__")
    return df.loc[:, columns].astype("string").fillna("__NA__").agg("\x1f".join, axis=1)


def fit_target_encoder(df: pd.DataFrame, spec: TargetEncodingSpec, prior: float) -> dict:
    key = make_group_key(df, spec.columns)
    stats = df.groupby(key, sort=False)[TARGET_COL].agg(["sum", "count"])
    values = (stats["sum"] + spec.smoothing * prior) / (stats["count"] + spec.smoothing)
    return values.to_dict()


def transform_target_encoder(df: pd.DataFrame, spec: TargetEncodingSpec, encoder: dict, prior: float) -> pd.Series:
    key = make_group_key(df, spec.columns)
    return key.map(encoder).astype("float64").fillna(prior)


def make_oof_target_encoding(
    df: pd.DataFrame,
    specs: list[TargetEncodingSpec],
    prior: float,
    n_splits: int = 5,
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for spec in specs:
        encoded[spec.name] = np.nan

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y = df[TARGET_COL].to_numpy()
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(df, y), start=1):
        fit_part = df.iloc[fit_idx]
        valid_part = df.iloc[valid_idx]
        for spec in specs:
            encoder = fit_target_encoder(fit_part, spec, prior)
            encoded.iloc[valid_idx, encoded.columns.get_loc(spec.name)] = transform_target_encoder(
                valid_part,
                spec,
                encoder,
                prior,
            ).to_numpy()
        print(f"  OOF target encoding fold {fold}/{n_splits} done")

    return encoded.fillna(prior)


def fit_full_target_encoders(
    df: pd.DataFrame,
    specs: list[TargetEncodingSpec],
    prior: float,
) -> dict[str, dict]:
    return {spec.name: fit_target_encoder(df, spec, prior) for spec in specs}


def apply_target_encoders(
    df: pd.DataFrame,
    specs: list[TargetEncodingSpec],
    encoders: dict[str, dict],
    prior: float,
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for spec in specs:
        encoded[spec.name] = transform_target_encoder(df, spec, encoders[spec.name], prior)
    return encoded


def build_features(df: pd.DataFrame, encoded: pd.DataFrame | None = None) -> pd.DataFrame:
    x = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").reset_index(drop=True).copy()
    if encoded is not None:
        x = pd.concat([x, encoded.reset_index(drop=True)], axis=1)
    return x


def make_pipeline(feature_columns: list[str]) -> Pipeline:
    cat_cols = [col for col in CAT_COLS if col in feature_columns]
    num_cols = [col for col in feature_columns if col not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ],
        remainder="drop",
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=320,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=250,
        l2_regularization=0.2,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def season_sample_weight(df: pd.DataFrame) -> np.ndarray:
    return df["season"].map(SEASON_WEIGHTS).fillna(1.0).astype("float64").to_numpy()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def apply_calibration(pred: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    return sigmoid(calibration["scale"] * logit(pred) + calibration["bias"])


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
    bss_score = max(0, 100000 * (1 - brier / mean_control_brier))
    return {
        "auc": float(roc_auc_score(y_true, pred)),
        "logloss": float(log_loss(y_true, pred)),
        "brier": float(brier),
        "bss_score": float(bss_score),
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


def train_and_validate(train: pd.DataFrame, prior: float) -> tuple[dict[str, float], dict[str, dict], list[str]]:
    fit_df = train[train["season"] <= 2023].copy()
    valid_df = train[train["season"] == 2024].copy()
    if fit_df.empty or valid_df.empty:
        raise ValueError("Need both <=2023 fit rows and 2024 validation rows for calibration.")

    print("Build OOF target encoding for validation fit data...")
    fit_oof = make_oof_target_encoding(fit_df, TE_SPECS, prior)
    fit_encoders = fit_full_target_encoders(fit_df, TE_SPECS, prior)
    valid_encoded = apply_target_encoders(valid_df, TE_SPECS, fit_encoders, prior)

    x_fit = build_features(fit_df, fit_oof)
    x_valid = build_features(valid_df, valid_encoded)
    model = make_pipeline(list(x_fit.columns))

    start = time.time()
    model.fit(x_fit, fit_df[TARGET_COL], clf__sample_weight=season_sample_weight(fit_df))
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

    return calibration, {"raw": raw_metrics, "calibrated": calibrated_metrics}, list(x_fit.columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=default_train_path())
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    args = parser.parse_args()

    print(f"Load train: {args.train_path}")
    train = load_train(args.train_path)
    prior = float(train[TARGET_COL].mean())
    print(f"Train rows={len(train)} target_mean={prior:.6f}")

    calibration, validation_metrics, _ = train_and_validate(train, prior)

    print("Build OOF target encoding for final train data...")
    final_oof = make_oof_target_encoding(train, TE_SPECS, prior)
    final_encoders = fit_full_target_encoders(train, TE_SPECS, prior)

    print("Build final features...")
    x_train = build_features(train, final_oof)
    model = make_pipeline(list(x_train.columns))

    start = time.time()
    model.fit(x_train, train[TARGET_COL], clf__sample_weight=season_sample_weight(train))
    elapsed = time.time() - start
    print(f"Final fit rows={len(x_train)} features={x_train.shape[1]} seconds={elapsed:.1f}")

    artifact = {
        "model": model,
        "feature_columns": list(x_train.columns),
        "target_encoders": final_encoders,
        "target_encoding_specs": [
            {"name": spec.name, "columns": spec.columns, "smoothing": spec.smoothing}
            for spec in TE_SPECS
        ],
        "target_prior": prior,
        "season_weights": SEASON_WEIGHTS,
        "calibration": calibration,
        "validation_metrics": validation_metrics,
        "id_col": ID_COL,
        "target_col": TARGET_COL,
        "sklearn_version": sklearn.__version__,
    }

    os.makedirs(args.model_path.parent, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
