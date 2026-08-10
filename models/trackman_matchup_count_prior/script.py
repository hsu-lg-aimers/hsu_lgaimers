import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
MODEL_FILENAME = "model.pkl"


def first_existing(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    data_dir = first_existing([
        cwd / "data",
        script_dir / "data",
        script_dir.parents[1] / "data",
    ])
    model_path = first_existing([
        cwd / "model" / MODEL_FILENAME,
        script_dir / "model" / MODEL_FILENAME,
    ])
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


def build_features(df: pd.DataFrame, prior: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    x = df.drop(columns=[TARGET_COL], errors="ignore").copy()
    x = x.merge(prior, on=group_cols, how="left")
    return x.drop(columns=[ID_COL], errors="ignore")


def align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_columns]


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


def main() -> None:
    test_path, sample_path, model_path, output_path = resolve_paths()

    print(f"Load model: {model_path}")
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    prior = artifact["trackman_prior"]
    group_cols = artifact["trackman_group_cols"]

    print(f"Load test: {test_path}")
    test = load_test(test_path)
    sub = load_sample_submission(sample_path)
    ids = test[ID_COL].tolist()
    print(f"test={len(test)} submission={len(sub)}")

    print("Build features...")
    x_test = build_features(test, prior, group_cols)
    x_test = align_features(x_test, feature_columns)
    print(f"features={x_test.shape[1]}")

    print("Inference model...")
    preds = model.predict_proba(x_test)[:, 1] if len(x_test) else np.array([])
    print(f"preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(output_path, sub)
    print(f"Saved: {output_path} rows={len(sub)}")


if __name__ == "__main__":
    main()
