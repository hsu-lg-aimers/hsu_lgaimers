import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
from scipy.optimize import minimize, minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
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
    "pitcher_exp_bin",
    "batter_exp_bin",
    "pressure_bin",
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

SMOOTHING_STRENGTHS = [50, 200, 800]

TRACKMAN_GROUP_COLS = [
    "pitcher_hand",
    "batter_hand",
    "balls_before",
    "strikes_before",
    "outs_before",
]
TRACKMAN_NUMERIC_COLS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
HAND_MAP = {"Left": 1, "Right": 2}
TRACKMAN_RATE_SMOOTHING_STRENGTHS = [100, 500, 1500]
DEFAULT_BLEND_WEIGHTS = {"lgbm": 0.12, "catboost": 0.78, "catboost_alt": 0.10}
META_MAX_ABS_DELTA = 0.060
META_DEFAULT_STRENGTH = 0.35
META_CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "count_state",
    "hand_matchup",
    "pitcher_exp_bin",
    "batter_exp_bin",
    "pressure_bin",
]
META_NUM_COLS = [
    "balls_before",
    "strikes_before",
    "outs_before",
    "inning",
    "li",
    "num_runners_on",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "score_diff_pitcher_team",
    "score_diff_home",
    "home_win_expectancy",
    "away_win_expectancy",
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_success_rate",
    "asof_batter_success_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
    "pitcher_recent5_success_delta",
    "pitcher_recent3_success_delta",
    "pitcher_recent1_success_delta",
    "pitcher_ball_minus_strike_rate",
    "batter_success_minus_pitcher_success",
    "fastball_minus_breaking_rate",
    "offspeed_share_of_nonfastball",
    "tm_state_n",
    "tm_state_rel_speed_mean",
    "tm_state_spin_rate_mean",
    "tm_state_induced_vert_break_mean",
    "tm_state_horz_break_mean",
    "tm_state_extension_mean",
    "tm_state_zone_speed_mean",
    "tm_state_fastball_rate_smooth_500",
    "tm_state_breaking_rate_smooth_500",
    "tm_state_offspeed_rate_smooth_500",
]


def find_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, Path.cwd(), *script_dir.parents, *Path.cwd().parents]
    seen = set()
    for base in candidates:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        if (base / "data" / "train.csv").exists():
            return base
    return Path.cwd()


def default_train_path() -> Path:
    return find_repo_root() / "data" / "train.csv"


def default_trackman_path() -> Path:
    return find_repo_root() / "data" / "trackman_history.csv"


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "model" / MODEL_FILENAME


def load_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {ID_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"train data is missing required columns: {sorted(missing)}")
    return df


def load_trackman(path: Path) -> pd.DataFrame:
    usecols = ["season"] + TRACKMAN_GROUP_COLS + TRACKMAN_NUMERIC_COLS + ["pitch_type_group"]
    df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    return df.dropna(subset=TRACKMAN_GROUP_COLS)


def build_trackman_prior(trackman: pd.DataFrame) -> pd.DataFrame:
    grouped = trackman.groupby(TRACKMAN_GROUP_COLS, dropna=False)

    global_pitch_rates = trackman["pitch_type_group"].value_counts(normalize=True)
    global_pitch_rates = {
        pitch_group: float(global_pitch_rates.get(pitch_group, 0.0))
        for pitch_group in PITCH_GROUPS
    }

    prior = grouped[TRACKMAN_NUMERIC_COLS].agg(["mean"]).reset_index()
    prior.columns = [
        "_".join([str(part) for part in col if part])
        for col in prior.columns.to_flat_index()
    ]
    rename_map = {}
    for col in prior.columns:
        if col not in TRACKMAN_GROUP_COLS:
            rename_map[col] = f"tm_state_{col}"
    prior = prior.rename(columns=rename_map)

    counts = grouped.size().reset_index(name="tm_state_n")
    prior = prior.merge(counts, on=TRACKMAN_GROUP_COLS, how="left")

    pitch_counts = (
        trackman.groupby(TRACKMAN_GROUP_COLS + ["pitch_type_group"], dropna=False)
        .size()
        .unstack("pitch_type_group", fill_value=0)
    )
    for pitch_group in PITCH_GROUPS:
        if pitch_group not in pitch_counts.columns:
            pitch_counts[pitch_group] = 0
    pitch_rates = pitch_counts[PITCH_GROUPS].div(pitch_counts[PITCH_GROUPS].sum(axis=1), axis=0)
    pitch_rates = pitch_rates.add_prefix("tm_state_").add_suffix("_rate").reset_index()

    prior = prior.merge(pitch_rates, on=TRACKMAN_GROUP_COLS, how="left")
    for strength in TRACKMAN_RATE_SMOOTHING_STRENGTHS:
        prior[f"tm_state_reliability_{strength}"] = prior["tm_state_n"] / (
            prior["tm_state_n"] + strength
        )
        for pitch_group in PITCH_GROUPS:
            rate_col = f"tm_state_{pitch_group}_rate"
            smooth_col = f"tm_state_{pitch_group}_rate_smooth_{strength}"
            prior[smooth_col] = (
                prior["tm_state_n"] * prior[rate_col]
                + strength * global_pitch_rates[pitch_group]
            ) / (prior["tm_state_n"] + strength)
    feature_cols = [col for col in prior.columns if col not in TRACKMAN_GROUP_COLS]
    prior[feature_cols] = prior[feature_cols].replace([np.inf, -np.inf], np.nan)
    prior.columns = pd.Index([str(col) for col in prior.columns], dtype=object)
    prior.index = pd.RangeIndex(len(prior))
    return prior


def serialize_trackman_prior(prior: pd.DataFrame) -> dict[str, object]:
    safe_prior = prior.copy()
    safe_prior.columns = pd.Index([str(col) for col in safe_prior.columns], dtype=object)
    safe_prior.index = pd.RangeIndex(len(safe_prior))
    return {
        "columns": [str(col) for col in safe_prior.columns],
        "records": safe_prior.astype(object).where(pd.notna(safe_prior), None).values.tolist(),
    }


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


def build_features(
    df: pd.DataFrame,
    rate_priors: dict[str, float],
    trackman_prior: pd.DataFrame | None = None,
) -> pd.DataFrame:
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

    x["score_abs_pitcher_team"] = x["score_diff_pitcher_team"].abs()
    x["score_abs_home"] = x["score_diff_home"].abs()
    x["pressure"] = x["li"] * (1 + x["num_runners_on"])
    x["late_inning"] = (x["inning"] >= 7).astype("int8")
    x["early_inning"] = (x["inning"] <= 3).astype("int8")

    x["pitcher_team_win_expectancy"] = np.where(
        x["top_bottom"].eq("T"),
        x["home_win_expectancy"],
        x["away_win_expectancy"],
    )
    x["batter_team_win_expectancy"] = 100 - x["pitcher_team_win_expectancy"]

    x["log_pitcher_n"] = np.log1p(x["asof_pitcher_n"].clip(lower=0))
    x["log_batter_n"] = np.log1p(x["asof_batter_n"].clip(lower=0))
    x["log_pitchmix_n"] = np.log1p(x["asof_pitcher_pitchmix_n"].clip(lower=0))
    x["pitcher_exp_bin"] = pd.cut(
        pd.to_numeric(x["asof_pitcher_n"], errors="coerce").fillna(0),
        bins=[-1, 10, 50, 150, 400, np.inf],
        labels=["p_000_010", "p_011_050", "p_051_150", "p_151_400", "p_401_plus"],
    ).astype(str)
    x["batter_exp_bin"] = pd.cut(
        pd.to_numeric(x["asof_batter_n"], errors="coerce").fillna(0),
        bins=[-1, 10, 50, 150, 400, np.inf],
        labels=["b_000_010", "b_011_050", "b_051_150", "b_151_400", "b_401_plus"],
    ).astype(str)
    x["pressure_bin"] = pd.cut(
        pd.to_numeric(x["pressure"], errors="coerce").fillna(0),
        bins=[-np.inf, 0.8, 1.5, 3.0, np.inf],
        labels=["low", "mid", "high", "extreme"],
    ).astype(str)

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

    if trackman_prior is not None:
        x = x.merge(trackman_prior, on=TRACKMAN_GROUP_COLS, how="left")
        x["fastball_minus_tm_state_fastball_rate"] = (
            x["asof_pitcher_fastball_rate"] - x["tm_state_fastball_rate"]
        )
        x["breaking_minus_tm_state_breaking_rate"] = (
            x["asof_pitcher_breaking_rate"] - x["tm_state_breaking_rate"]
        )
        x["offspeed_minus_tm_state_offspeed_rate"] = (
            x["asof_pitcher_offspeed_rate"] - x["tm_state_offspeed_rate"]
        )
        x["pitcher_speed_context"] = x["asof_pitcher_fastball_rate"] * x["tm_state_rel_speed_mean"]

    return x


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    cat_cols = [col for col in CAT_COLS if col in feature_columns]
    num_cols = [col for col in feature_columns if col not in cat_cols]

    return ColumnTransformer(
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


def make_lgbm_pipeline(feature_columns: list[str], lgbm_device: str = "cpu") -> Pipeline:
    params = {
        "objective": "regression",
        "metric": "l2",
        "n_estimators": 900,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_child_samples": 140,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 0.20,
        "reg_alpha": 0.02,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if lgbm_device == "gpu":
        params.update({"device_type": "gpu", "gpu_use_dp": False})

    return Pipeline([("pre", make_preprocessor(feature_columns)), ("reg", LGBMRegressor(**params))])


def make_catboost_model(
    catboost_device: str = "cpu",
    gpu_device: str = "0",
    variant: str = "base",
) -> CatBoostClassifier:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 900,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": 43,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
    }
    if variant == "alt":
        params.update(
            {
                "iterations": 1200,
                "learning_rate": 0.028,
                "depth": 8,
                "l2_leaf_reg": 8.0,
                "random_seed": 202,
            }
        )
    elif variant != "base":
        raise ValueError(f"unknown catboost variant: {variant}")
    if catboost_device == "gpu":
        params.update({"task_type": "GPU", "devices": gpu_device})
    return CatBoostClassifier(**params)


def prepare_catboost_frame(
    x: pd.DataFrame,
    feature_columns: list[str],
    cat_columns: list[str],
) -> pd.DataFrame:
    x_cb = x.loc[:, feature_columns].copy()
    for col in cat_columns:
        values = x_cb[col].to_numpy(dtype=object, copy=False)
        x_cb[col] = np.where(pd.isna(values), "__MISSING__", values).astype(str)
    return x_cb


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def apply_calibration(pred: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    return sigmoid(calibration["scale"] * logit(pred) + calibration["bias"])


def make_meta_frame(
    x: pd.DataFrame,
    lgbm_pred: np.ndarray,
    cat_pred: np.ndarray,
    cat_alt_pred: np.ndarray,
    raw_pred: np.ndarray,
    calibrated_pred: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    meta = pd.DataFrame(index=x.index)
    meta["meta_lgbm_pred"] = lgbm_pred
    meta["meta_catboost_pred"] = cat_pred
    meta["meta_catboost_alt_pred"] = cat_alt_pred
    meta["meta_raw_pred"] = raw_pred
    meta["meta_calibrated_pred"] = calibrated_pred
    meta["meta_lgbm_logit"] = logit(lgbm_pred)
    meta["meta_catboost_logit"] = logit(cat_pred)
    meta["meta_catboost_alt_logit"] = logit(cat_alt_pred)
    meta["meta_raw_logit"] = logit(raw_pred)
    meta["meta_calibrated_logit"] = logit(calibrated_pred)
    meta["meta_cat_minus_lgbm"] = cat_pred - lgbm_pred
    meta["meta_cat_alt_minus_cat"] = cat_alt_pred - cat_pred
    meta["meta_model_spread"] = np.maximum.reduce([lgbm_pred, cat_pred, cat_alt_pred]) - np.minimum.reduce(
        [lgbm_pred, cat_pred, cat_alt_pred]
    )

    cat_cols = []
    for col in META_CAT_COLS:
        if col in x.columns:
            meta[col] = x[col].astype(object).where(pd.notna(x[col]), "__MISSING__").astype(str)
            cat_cols.append(col)

    num_cols = [
        "meta_lgbm_pred",
        "meta_catboost_pred",
        "meta_catboost_alt_pred",
        "meta_raw_pred",
        "meta_calibrated_pred",
        "meta_lgbm_logit",
        "meta_catboost_logit",
        "meta_catboost_alt_logit",
        "meta_raw_logit",
        "meta_calibrated_logit",
        "meta_cat_minus_lgbm",
        "meta_cat_alt_minus_cat",
        "meta_model_spread",
    ]
    for col in META_NUM_COLS:
        if col in x.columns:
            meta[col] = pd.to_numeric(x[col], errors="coerce")
            num_cols.append(col)

    return meta, cat_cols, num_cols


def make_meta_pipeline(cat_cols: list[str], num_cols: list[str]) -> Pipeline:
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
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.035,
        max_iter=260,
        max_leaf_nodes=15,
        min_samples_leaf=300,
        l2_regularization=0.08,
        random_state=45,
    )
    return Pipeline([("pre", preprocessor), ("hgb", model)])


def fit_meta_residual_corrector(
    x: pd.DataFrame,
    lgbm_pred: np.ndarray,
    cat_pred: np.ndarray,
    cat_alt_pred: np.ndarray,
    raw_pred: np.ndarray,
    calibrated_pred: np.ndarray,
    y: pd.Series,
) -> tuple[dict[str, object], np.ndarray]:
    meta_x, cat_cols, num_cols = make_meta_frame(x, lgbm_pred, cat_pred, cat_alt_pred, raw_pred, calibrated_pred)
    y_values = y.to_numpy(dtype="float64")
    residual = y_values - calibrated_pred
    model = make_meta_pipeline(cat_cols, num_cols)
    model.fit(meta_x, residual)
    raw_delta = np.clip(model.predict(meta_x), -META_MAX_ABS_DELTA, META_MAX_ABS_DELTA)

    def objective(strength: float) -> float:
        adjusted = np.clip(calibrated_pred + strength * raw_delta, 1e-6, 1 - 1e-6)
        return float(np.mean((adjusted - y_values) ** 2))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
    strength = float(result.x if result.success else META_DEFAULT_STRENGTH)
    adjusted = np.clip(calibrated_pred + strength * raw_delta, 1e-6, 1 - 1e-6)
    print(
        "  meta_residual_hgb: "
        f"features={meta_x.shape[1]} "
        f"cat={len(cat_cols)} "
        f"num={len(num_cols)} "
        f"strength={strength:.6f} "
        f"max_abs_delta={META_MAX_ABS_DELTA:.3f}"
    )
    return (
        {
            "enabled": True,
            "model": model,
            "cat_columns": cat_cols,
            "num_columns": num_cols,
            "feature_columns": list(meta_x.columns),
            "strength": strength,
            "max_abs_delta": META_MAX_ABS_DELTA,
            "target": "residual_after_global_calibration",
        },
        adjusted,
    )


def apply_meta_residual_corrector(
    x: pd.DataFrame,
    lgbm_pred: np.ndarray,
    cat_pred: np.ndarray,
    cat_alt_pred: np.ndarray,
    raw_pred: np.ndarray,
    calibrated_pred: np.ndarray,
    corrector: dict[str, object] | None,
) -> np.ndarray:
    if not corrector or not corrector.get("enabled"):
        return calibrated_pred
    meta_x, _, _ = make_meta_frame(x, lgbm_pred, cat_pred, cat_alt_pred, raw_pred, calibrated_pred)
    feature_columns = corrector.get("feature_columns", list(meta_x.columns))
    for col in feature_columns:
        if col not in meta_x.columns:
            meta_x[col] = np.nan
    meta_x = meta_x.loc[:, feature_columns]
    max_abs_delta = float(corrector.get("max_abs_delta", META_MAX_ABS_DELTA))
    strength = float(corrector.get("strength", META_DEFAULT_STRENGTH))
    delta = np.clip(corrector["model"].predict(meta_x), -max_abs_delta, max_abs_delta)
    return np.clip(calibrated_pred + strength * delta, 1e-6, 1 - 1e-6)


def strip_predict_unused_rng(model: Pipeline) -> None:
    estimator = model.named_steps.get("reg")
    if estimator is not None and hasattr(estimator, "_feature_subsample_rng"):
        estimator._feature_subsample_rng = None


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


def fit_blend_weight(lgbm_pred: np.ndarray, cat_pred: np.ndarray, y: pd.Series) -> float:
    y_values = y.to_numpy(dtype="float64")

    def objective(weight: float) -> float:
        pred = weight * lgbm_pred + (1 - weight) * cat_pred
        return float(np.mean((pred - y_values) ** 2))

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-5})
    return float(result.x)


def fit_blend_weights_3way(
    lgbm_pred: np.ndarray,
    cat_pred: np.ndarray,
    cat_alt_pred: np.ndarray,
    y: pd.Series,
) -> dict[str, float]:
    y_values = y.to_numpy(dtype="float64")
    pred_matrix = np.vstack([lgbm_pred, cat_pred, cat_alt_pred]).T

    def objective(weights: np.ndarray) -> float:
        pred = pred_matrix @ weights
        return float(np.mean((pred - y_values) ** 2))

    result = minimize(
        objective,
        x0=np.array(
            [
                DEFAULT_BLEND_WEIGHTS["lgbm"],
                DEFAULT_BLEND_WEIGHTS["catboost"],
                DEFAULT_BLEND_WEIGHTS["catboost_alt"],
            ]
        ),
        method="SLSQP",
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 300, "ftol": 1e-12},
    )
    weights = result.x if result.success else np.array(
        [
            DEFAULT_BLEND_WEIGHTS["lgbm"],
            DEFAULT_BLEND_WEIGHTS["catboost"],
            DEFAULT_BLEND_WEIGHTS["catboost_alt"],
        ]
    )
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    return {
        "lgbm": float(weights[0]),
        "catboost": float(weights[1]),
        "catboost_alt": float(weights[2]),
        "success": bool(result.success),
        "objective": float(objective(weights)),
    }


def blend_predictions(lgbm_pred: np.ndarray, cat_pred: np.ndarray, lgbm_weight: float) -> np.ndarray:
    return np.clip(lgbm_weight * lgbm_pred + (1 - lgbm_weight) * cat_pred, 0, 1)


def blend_predictions_3way(
    lgbm_pred: np.ndarray,
    cat_pred: np.ndarray,
    cat_alt_pred: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    weights = {
        "lgbm": float(weights.get("lgbm", DEFAULT_BLEND_WEIGHTS["lgbm"])),
        "catboost": float(weights.get("catboost", DEFAULT_BLEND_WEIGHTS["catboost"])),
        "catboost_alt": float(weights.get("catboost_alt", DEFAULT_BLEND_WEIGHTS["catboost_alt"])),
    }
    total = sum(weights.values())
    if total <= 0:
        weights = dict(DEFAULT_BLEND_WEIGHTS)
        total = sum(weights.values())
    weights = {key: value / total for key, value in weights.items()}
    pred = (
        weights["lgbm"] * lgbm_pred
        + weights["catboost"] * cat_pred
        + weights["catboost_alt"] * cat_alt_pred
    )
    return np.clip(pred, 0, 1)


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


def fit_ensemble(
    x_fit: pd.DataFrame,
    y_fit: pd.Series,
    feature_columns: list[str],
    cat_columns: list[str],
    sample_weight: np.ndarray | None = None,
    lgbm_device: str = "cpu",
    catboost_device: str = "cpu",
    gpu_device: str = "0",
) -> dict[str, object]:
    lgbm_model = make_lgbm_pipeline(feature_columns, lgbm_device)
    cat_model = make_catboost_model(catboost_device, gpu_device, "base")
    cat_alt_model = make_catboost_model(catboost_device, gpu_device, "alt")

    start = time.time()
    print(f"  LightGBM device={lgbm_device}")
    if sample_weight is None:
        lgbm_model.fit(x_fit, y_fit)
    else:
        lgbm_model.fit(x_fit, y_fit, reg__sample_weight=sample_weight)
    print(f"  LightGBM fit seconds={time.time() - start:.1f}")

    x_fit_cb = prepare_catboost_frame(x_fit, feature_columns, cat_columns)
    start = time.time()
    print(f"  CatBoost device={catboost_device} gpu_device={gpu_device if catboost_device == 'gpu' else '-'}")
    cat_model.fit(x_fit_cb, y_fit, cat_features=cat_columns, sample_weight=sample_weight)
    print(f"  CatBoost fit seconds={time.time() - start:.1f}")

    start = time.time()
    print(f"  CatBoost alt device={catboost_device} gpu_device={gpu_device if catboost_device == 'gpu' else '-'}")
    cat_alt_model.fit(x_fit_cb, y_fit, cat_features=cat_columns, sample_weight=sample_weight)
    print(f"  CatBoost alt fit seconds={time.time() - start:.1f}")

    return {"lgbm": lgbm_model, "catboost": cat_model, "catboost_alt": cat_alt_model}


def predict_ensemble_raw(
    models: dict[str, object],
    x: pd.DataFrame,
    feature_columns: list[str],
    cat_columns: list[str],
    blend_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lgbm_pred = np.clip(models["lgbm"].predict(x.loc[:, feature_columns]), 0, 1)
    x_cb = prepare_catboost_frame(x, feature_columns, cat_columns)
    cat_pred = models["catboost"].predict_proba(x_cb)[:, 1]
    cat_alt_pred = models["catboost_alt"].predict_proba(x_cb)[:, 1]
    ensemble_pred = blend_predictions_3way(lgbm_pred, cat_pred, cat_alt_pred, blend_weights)
    return lgbm_pred, cat_pred, cat_alt_pred, ensemble_pred


def train_and_validate(
    train: pd.DataFrame,
    trackman: pd.DataFrame,
    variant: str,
    validation_pred_path: Path | None = None,
    lgbm_device: str = "cpu",
    catboost_device: str = "cpu",
    gpu_device: str = "0",
) -> tuple[dict[str, float], dict[str, float], dict[str, object], dict[str, dict]]:
    if train["season"].nunique() < 2 or 2024 not in set(train["season"]):
        print("Skip holdout validation: season 2024 is not available.")
        return (
            {**DEFAULT_BLEND_WEIGHTS, "success": True, "objective": np.nan},
            {"scale": 1.0, "bias": 0.0, "success": True, "objective": np.nan},
            {"meta_residual": {"enabled": False}},
            {},
        )

    fit_df = train[train["season"] <= 2023].copy()
    valid_df = train[train["season"] == 2024].copy()
    if fit_df.empty or valid_df.empty:
        print("Skip holdout validation: not enough rows.")
        return (
            {**DEFAULT_BLEND_WEIGHTS, "success": True, "objective": np.nan},
            {"scale": 1.0, "bias": 0.0, "success": True, "objective": np.nan},
            {"meta_residual": {"enabled": False}},
            {},
        )

    sample_weight = None
    if variant == "drop_2023":
        fit_df = fit_df[fit_df["season"] != 2023].copy()
    elif variant == "weight_2023_050":
        sample_weight = np.where(fit_df["season"].to_numpy() == 2023, 0.5, 1.0)
    elif variant == "weight_2023_030":
        sample_weight = np.where(fit_df["season"].to_numpy() == 2023, 0.3, 1.0)
    elif variant == "drop_2023_f":
        fit_df = fit_df[~((fit_df["season"] == 2023) & (fit_df["game_type"] == "F"))].copy()
    elif variant == "drop_pre2023_f":
        fit_df = fit_df[~((fit_df["season"] <= 2022) & (fit_df["game_type"] == "F"))].copy()
    elif variant == "drop_through2023_f":
        fit_df = fit_df[fit_df["game_type"] != "F"].copy()
    elif variant != "baseline":
        raise ValueError(f"unknown variant: {variant}")

    rate_priors = make_rate_priors(fit_df)
    validation_trackman_prior = build_trackman_prior(trackman[trackman["season"] <= 2023].copy())
    print(
        f"Validation Trackman prior groups={len(validation_trackman_prior)} "
        f"features={validation_trackman_prior.shape[1] - len(TRACKMAN_GROUP_COLS)}"
    )
    x_fit = build_features(fit_df, rate_priors, validation_trackman_prior)
    x_valid = build_features(valid_df, rate_priors, validation_trackman_prior)
    feature_columns = list(x_fit.columns)
    cat_columns = [col for col in CAT_COLS if col in feature_columns]

    print(
        f"Validation variant={variant} "
        f"fit rows={len(x_fit)} valid rows={len(x_valid)} features={len(feature_columns)}"
    )
    models = fit_ensemble(
        x_fit,
        fit_df[TARGET_COL],
        feature_columns,
        cat_columns,
        sample_weight,
        lgbm_device,
        catboost_device,
        gpu_device,
    )

    lgbm_pred = np.clip(models["lgbm"].predict(x_valid.loc[:, feature_columns]), 0, 1)
    x_valid_cb = prepare_catboost_frame(x_valid, feature_columns, cat_columns)
    cat_pred = models["catboost"].predict_proba(x_valid_cb)[:, 1]
    cat_alt_pred = models["catboost_alt"].predict_proba(x_valid_cb)[:, 1]
    blend_weights = fit_blend_weights_3way(lgbm_pred, cat_pred, cat_alt_pred, valid_df[TARGET_COL])
    raw_pred = blend_predictions_3way(lgbm_pred, cat_pred, cat_alt_pred, blend_weights)

    calibration = fit_logit_brier_calibration(raw_pred, valid_df[TARGET_COL])
    calibrated_pred = apply_calibration(raw_pred, calibration)
    meta_residual, meta_pred = fit_meta_residual_corrector(
        x_valid,
        lgbm_pred,
        cat_pred,
        cat_alt_pred,
        raw_pred,
        calibrated_pred,
        valid_df[TARGET_COL],
    )
    postprocess = {
        "meta_residual": meta_residual,
        "order": ["global_logit_calibration", "meta_residual_hgb"],
    }

    if validation_pred_path is not None:
        pred_df = pd.DataFrame(
            {
                ID_COL: valid_df[ID_COL].to_numpy(),
                TARGET_COL: valid_df[TARGET_COL].to_numpy(),
                "season": valid_df["season"].to_numpy(),
                "game_month": valid_df["game_month"].to_numpy(),
                "game_type": valid_df["game_type"].to_numpy(),
                "balls_before": valid_df["balls_before"].to_numpy(),
                "strikes_before": valid_df["strikes_before"].to_numpy(),
                "outs_before": valid_df["outs_before"].to_numpy(),
                "pitcher_hand": valid_df["pitcher_hand"].to_numpy(),
                "batter_hand": valid_df["batter_hand"].to_numpy(),
                "asof_pitcher_n": valid_df["asof_pitcher_n"].to_numpy(),
                "asof_batter_n": valid_df["asof_batter_n"].to_numpy(),
                "lgbm_pred": lgbm_pred,
                "catboost_pred": cat_pred,
                "catboost_alt_pred": cat_alt_pred,
                "ensemble_raw": raw_pred,
                "ensemble_calibrated": calibrated_pred,
                "ensemble_meta_residual": meta_pred,
            }
        )
        os.makedirs(validation_pred_path.parent, exist_ok=True)
        pred_df.to_csv(validation_pred_path, index=False, encoding="utf-8")
        print(f"Saved validation predictions: {validation_pred_path} rows={len(pred_df)}")

    metrics = {
        "lgbm_raw": score_dict(valid_df[TARGET_COL], lgbm_pred),
        "catboost_raw": score_dict(valid_df[TARGET_COL], cat_pred),
        "catboost_alt_raw": score_dict(valid_df[TARGET_COL], cat_alt_pred),
        "ensemble_raw": score_dict(valid_df[TARGET_COL], raw_pred),
        "ensemble_calibrated": score_dict(valid_df[TARGET_COL], calibrated_pred),
        "ensemble_meta_residual": score_dict(valid_df[TARGET_COL], meta_pred),
    }
    metrics["variant"] = {
        "name": variant,
        "fit_rows": int(len(fit_df)),
        "sample_weight_used": sample_weight is not None,
    }

    print(
        "blend: "
        f"lgbm_weight={blend_weights['lgbm']:.6f} "
        f"catboost_weight={blend_weights['catboost']:.6f} "
        f"catboost_alt_weight={blend_weights['catboost_alt']:.6f} "
        f"success={blend_weights['success']}"
    )
    for name, score in metrics.items():
        if isinstance(score, dict) and "auc" in score:
            print_metrics(f"holdout_2024_{name}", score)
    print(
        "calibration: "
        f"scale={calibration['scale']:.8f} "
        f"bias={calibration['bias']:.8f} "
        f"success={calibration['success']}"
    )

    return blend_weights, calibration, postprocess, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=default_train_path())
    parser.add_argument("--trackman-path", type=Path, default=default_trackman_path())
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--lgbm-weight", type=float)
    parser.add_argument("--calibration-scale", type=float)
    parser.add_argument("--calibration-bias", type=float)
    parser.add_argument("--validation-pred-path", type=Path)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--lgbm-device", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--catboost-device", choices=["cpu", "gpu"], default=None)
    parser.add_argument(
        "--variant",
        choices=[
            "baseline",
            "drop_2023",
            "weight_2023_050",
            "weight_2023_030",
            "drop_2023_f",
            "drop_pre2023_f",
            "drop_through2023_f",
        ],
        default="baseline",
    )
    args = parser.parse_args()
    lgbm_device = args.lgbm_device or ("gpu" if args.use_gpu else "cpu")
    catboost_device = args.catboost_device or ("gpu" if args.use_gpu else "cpu")
    print(
        "Training devices: "
        f"LightGBM={lgbm_device} "
        f"CatBoost={catboost_device} "
        f"CatBoostAlt={catboost_device} "
        f"gpu_device={args.gpu_device}"
    )

    print(f"Load train: {args.train_path}")
    train = load_train(args.train_path)
    rate_priors = make_rate_priors(train)
    print(f"Train rows={len(train)} target_mean={rate_priors[TARGET_COL]:.6f}")

    print(f"Load trackman: {args.trackman_path}")
    trackman = load_trackman(args.trackman_path)
    final_trackman_prior = build_trackman_prior(trackman)
    print(
        f"Final Trackman prior groups={len(final_trackman_prior)} "
        f"features={final_trackman_prior.shape[1] - len(TRACKMAN_GROUP_COLS)}"
    )

    blend_weights = {**DEFAULT_BLEND_WEIGHTS, "success": True, "objective": np.nan}
    calibration = {"scale": 1.0, "bias": 0.0, "success": True, "objective": np.nan}
    postprocess = {"meta_residual": {"enabled": False}, "order": ["global_logit_calibration", "meta_residual_hgb"]}
    validation_metrics = {}
    if not args.skip_validation:
        blend_weights, calibration, postprocess, validation_metrics = train_and_validate(
            train,
            trackman,
            args.variant,
            args.validation_pred_path,
            lgbm_device,
            catboost_device,
            args.gpu_device,
        )
        if args.validate_only:
            print("Validate-only mode: skip final fit and model save.")
            return
    elif args.lgbm_weight is not None:
        if args.calibration_scale is None or args.calibration_bias is None:
            raise ValueError("--skip-validation with --lgbm-weight also requires calibration scale and bias.")
        lgbm_weight = float(args.lgbm_weight)
        blend_weights = {
            "lgbm": lgbm_weight,
            "catboost": 1 - lgbm_weight,
            "catboost_alt": 0.0,
            "success": True,
            "objective": np.nan,
        }
        calibration = {
            "scale": float(args.calibration_scale),
            "bias": float(args.calibration_bias),
            "success": True,
            "objective": np.nan,
        }
        validation_metrics = {
            "provided": {
                "blend_weights": blend_weights,
                "calibration_scale": calibration["scale"],
                "calibration_bias": calibration["bias"],
            }
        }

    print("Build final features...")
    x_train = build_features(train, rate_priors, final_trackman_prior)
    feature_columns = list(x_train.columns)
    cat_columns = [col for col in CAT_COLS if col in feature_columns]
    models = fit_ensemble(
        x_train,
        train[TARGET_COL],
        feature_columns,
        cat_columns,
        None,
        lgbm_device,
        catboost_device,
        args.gpu_device,
    )
    strip_predict_unused_rng(models["lgbm"])

    artifact = {
        "models": models,
        "feature_columns": feature_columns,
        "cat_columns": cat_columns,
        "rate_priors": rate_priors,
        "trackman_prior": serialize_trackman_prior(final_trackman_prior),
        "trackman_group_cols": TRACKMAN_GROUP_COLS,
        "blend_weights": blend_weights,
        "lgbm_weight": blend_weights["lgbm"],
        "calibration": calibration,
        "postprocess": postprocess,
        "validation_metrics": validation_metrics,
        "training_devices": {
            "lgbm_device": lgbm_device,
            "catboost_device": catboost_device,
            "catboost_alt_device": catboost_device,
            "gpu_device": args.gpu_device,
        },
        "target_col": TARGET_COL,
        "id_col": ID_COL,
        "sklearn_version": sklearn.__version__,
    }

    os.makedirs(args.model_path.parent, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
