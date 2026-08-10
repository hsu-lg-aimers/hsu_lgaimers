import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_FILENAME = "model.pkl"

CAT_COLS = ["top_bottom", "game_type", "base_state"]
TRACKMAN_GROUP_COLS = ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"]
HAND_MAP = {"Left": 1, "Right": 2}

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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_train_path() -> Path:
    return repo_root() / "data" / "train.csv"


def default_trackman_path() -> Path:
    return repo_root() / "data" / "trackman_history.csv"


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "model" / MODEL_FILENAME


def load_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {ID_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"train data is missing required columns: {sorted(missing)}")
    return df


def load_trackman(path: Path) -> pd.DataFrame:
    usecols = TRACKMAN_GROUP_COLS + TRACKMAN_NUMERIC_COLS + ["pitch_type_group"]
    df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
    df["pitcher_hand"] = df["pitcher_hand"].map(HAND_MAP)
    df["batter_hand"] = df["batter_hand"].map(HAND_MAP)
    return df.dropna(subset=TRACKMAN_GROUP_COLS)


def build_trackman_prior(trackman: pd.DataFrame) -> pd.DataFrame:
    grouped = trackman.groupby(TRACKMAN_GROUP_COLS, dropna=False)

    prior = grouped[TRACKMAN_NUMERIC_COLS].agg(["mean", "std"]).reset_index()
    prior.columns = [
        "_".join([str(part) for part in col if part])
        for col in prior.columns.to_flat_index()
    ]
    prior = prior.rename(columns={col: col.replace("_mean", "_tm_mean").replace("_std", "_tm_std") for col in prior.columns})

    counts = grouped.size().reset_index(name="tm_matchup_count_n")
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
    pitch_rates = pitch_rates.add_prefix("tm_matchup_count_").add_suffix("_rate").reset_index()

    prior = prior.merge(pitch_rates, on=TRACKMAN_GROUP_COLS, how="left")

    rename_map = {}
    for col in prior.columns:
        if col in TRACKMAN_GROUP_COLS or col == "tm_matchup_count_n":
            continue
        if not col.startswith("tm_matchup_count_"):
            rename_map[col] = f"tm_matchup_count_{col}"
    prior = prior.rename(columns=rename_map)

    feature_cols = [col for col in prior.columns if col not in TRACKMAN_GROUP_COLS]
    prior[feature_cols] = prior[feature_cols].replace([np.inf, -np.inf], np.nan)
    return prior


def add_trackman_prior(df: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[TARGET_COL], errors="ignore").copy()
    x = x.merge(prior, on=TRACKMAN_GROUP_COLS, how="left")
    return x


def build_features(df: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    x = add_trackman_prior(df, prior)
    return x.drop(columns=[ID_COL], errors="ignore")


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

    classifier = RandomForestClassifier(
        n_estimators=240,
        max_depth=12,
        min_samples_leaf=200,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", preprocessor), ("clf", classifier)])


def print_metrics(name: str, y_true: pd.Series, pred: np.ndarray) -> None:
    pred = np.clip(pred, 1e-6, 1 - 1e-6)
    brier = brier_score_loss(y_true, pred)
    mean_control_rate = float(y_true.mean())
    mean_control_brier = mean_control_rate * (1 - mean_control_rate)
    competition_score = max(0, 100000 * (1 - brier / mean_control_brier))
    print(
        f"{name}: "
        f"auc={roc_auc_score(y_true, pred):.6f} "
        f"logloss={log_loss(y_true, pred):.6f} "
        f"brier={brier:.6f} "
        f"bss_score={competition_score:.6f} "
        f"mean_control_rate={mean_control_rate:.6f} "
        f"mean_control_brier={mean_control_brier:.6f} "
        f"ap={average_precision_score(y_true, pred):.6f} "
        f"mean_pred={pred.mean():.6f}"
    )


def train_and_validate(train: pd.DataFrame, prior: pd.DataFrame) -> None:
    if 2024 not in set(train["season"]):
        print("Skip holdout validation: season 2024 is not available.")
        return

    fit_df = train[train["season"] <= 2023].copy()
    valid_df = train[train["season"] == 2024].copy()
    if fit_df.empty or valid_df.empty:
        print("Skip holdout validation: not enough rows.")
        return

    x_fit = build_features(fit_df, prior)
    x_valid = build_features(valid_df, prior)
    model = make_pipeline(list(x_fit.columns))

    start = time.time()
    model.fit(x_fit, fit_df[TARGET_COL])
    elapsed = time.time() - start
    pred = model.predict_proba(x_valid)[:, 1]

    print(f"Validation fit rows={len(x_fit)} valid rows={len(x_valid)} seconds={elapsed:.1f}")
    print_metrics("holdout_2024", valid_df[TARGET_COL], pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=default_train_path())
    parser.add_argument("--trackman-path", type=Path, default=default_trackman_path())
    parser.add_argument("--model-path", type=Path, default=default_model_path())
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    print(f"Load train: {args.train_path}")
    train = load_train(args.train_path)
    print(f"Train rows={len(train)} target_mean={train[TARGET_COL].mean():.6f}")

    print(f"Load trackman: {args.trackman_path}")
    trackman = load_trackman(args.trackman_path)
    prior = build_trackman_prior(trackman)
    print(f"Trackman prior groups={len(prior)} features={prior.shape[1] - len(TRACKMAN_GROUP_COLS)}")

    if not args.skip_validation:
        train_and_validate(train, prior)

    print("Build final features...")
    x_train = build_features(train, prior)
    model = make_pipeline(list(x_train.columns))

    start = time.time()
    model.fit(x_train, train[TARGET_COL])
    elapsed = time.time() - start
    print(f"Final fit rows={len(x_train)} features={x_train.shape[1]} seconds={elapsed:.1f}")

    artifact = {
        "model": model,
        "feature_columns": list(x_train.columns),
        "trackman_prior": prior,
        "trackman_group_cols": TRACKMAN_GROUP_COLS,
        "id_col": ID_COL,
        "target_col": TARGET_COL,
        "sklearn_version": sklearn.__version__,
    }

    os.makedirs(args.model_path.parent, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
