import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_FILENAME = "model.pkl"

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


def first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def unique_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    base_dirs = unique_paths([cwd, script_dir, *cwd.parents, *script_dir.parents])
    data_dir = first_existing([base / "data" for base in base_dirs])
    model_path = first_existing([base / "model" / MODEL_FILENAME for base in base_dirs])
    output_dir = cwd / "output"

    return (
        data_dir / "test.csv",
        data_dir / "sample_submission.csv",
        model_path,
        output_dir / "submission.csv",
    )


def load_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test data is missing {ID_COL}: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"submission columns must start with {[ID_COL, TARGET_COL]}: {list(df.columns)}")
    return df


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
    trackman_group_cols: list[str] | None = None,
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
        group_cols = trackman_group_cols if trackman_group_cols is not None else TRACKMAN_GROUP_COLS
        x = x.merge(trackman_prior, on=group_cols, how="left")
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


def align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_columns]


def deserialize_trackman_prior(payload) -> pd.DataFrame | None:
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        prior = payload.copy()
    else:
        prior = pd.DataFrame(payload["records"], columns=payload["columns"])
    prior.columns = pd.Index([str(col) for col in prior.columns], dtype=object)
    prior.index = pd.RangeIndex(len(prior))
    return prior


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


def blend_predictions(lgbm_pred: np.ndarray, cat_pred: np.ndarray, lgbm_weight: float) -> np.ndarray:
    return np.clip(lgbm_weight * lgbm_pred + (1 - lgbm_weight) * cat_pred, 0, 1)


def blend_predictions_3way(
    lgbm_pred: np.ndarray,
    cat_pred: np.ndarray,
    et_pred: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    pred = (
        weights["lgbm"] * lgbm_pred
        + weights["catboost"] * cat_pred
        + weights["extratrees"] * et_pred
    )
    return np.clip(pred, 0, 1)


def merge_predictions(sub: pd.DataFrame, ids: list[str], preds: np.ndarray) -> pd.DataFrame:
    pred_map = dict(zip(ids, preds))
    values = []
    missing = 0
    for row_id, current_value in zip(sub[ID_COL], sub[TARGET_COL]):
        pred = pred_map.get(row_id)
        if pred is None:
            missing += 1
            values.append(current_value)
        else:
            values.append(float(np.clip(pred, 0, 1)))
    if missing:
        print(f"Warning: kept placeholder for {missing} rows without predictions.")
    sub[TARGET_COL] = values
    return sub


def save_submission(path: Path, sub: pd.DataFrame) -> None:
    os.makedirs(path.parent, exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def apply_calibration(pred: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    return sigmoid(calibration["scale"] * logit(pred) + calibration["bias"])


def main() -> None:
    test_path, sample_path, model_path, output_path = resolve_paths()

    print(f"Load model: {model_path}")
    artifact = joblib.load(model_path)
    models = artifact["models"]
    feature_columns = artifact["feature_columns"]
    cat_columns = artifact["cat_columns"]
    rate_priors = artifact["rate_priors"]
    trackman_prior = deserialize_trackman_prior(artifact.get("trackman_prior"))
    trackman_group_cols = artifact.get("trackman_group_cols")
    blend_weights = artifact.get("blend_weights")
    if blend_weights is None:
        lgbm_weight = artifact.get("lgbm_weight", 0.5)
        blend_weights = {
            "lgbm": lgbm_weight,
            "catboost": 1 - lgbm_weight,
            "extratrees": 0.0,
        }
    calibration = artifact.get("calibration", {"scale": 1.0, "bias": 0.0})

    print(f"Load test: {test_path}")
    test = load_test(test_path)
    sub = load_sample_submission(sample_path)
    ids = test[ID_COL].tolist()
    print(f"test={len(test)} submission={len(sub)}")

    print("Build features...")
    x_test = build_features(test, rate_priors, trackman_prior, trackman_group_cols)
    x_test = align_features(x_test, feature_columns)
    print(f"features={x_test.shape[1]}")

    print("Inference ensemble...")
    if len(x_test):
        lgbm_pred = np.clip(models["lgbm"].predict(x_test), 0, 1)
        x_test_cb = prepare_catboost_frame(x_test, feature_columns, cat_columns)
        cat_pred = models["catboost"].predict_proba(x_test_cb)[:, 1]
        if "extratrees" in models:
            et_pred = models["extratrees"].predict_proba(x_test)[:, 1]
        else:
            et_pred = np.zeros_like(lgbm_pred)
        preds = blend_predictions_3way(lgbm_pred, cat_pred, et_pred, blend_weights)
        preds = apply_calibration(preds, calibration)
    else:
        preds = np.array([])
    print(f"preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(output_path, sub)
    print(f"Saved: {output_path} rows={len(sub)}")


if __name__ == "__main__":
    main()
