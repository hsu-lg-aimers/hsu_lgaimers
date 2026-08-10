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


def make_group_key(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    if len(columns) == 1:
        return df[columns[0]].astype("string").fillna("__NA__")
    return df.loc[:, columns].astype("string").fillna("__NA__").agg("\x1f".join, axis=1)


def apply_target_encoders(
    df: pd.DataFrame,
    specs: list,
    encoders: dict[str, dict],
    prior: float,
) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)
    for spec in specs:
        name = spec["name"]
        columns = tuple(spec["columns"])
        key = make_group_key(df, columns)
        encoded[name] = key.map(encoders[name]).astype("float64").fillna(prior)
    return encoded


def build_features(df: pd.DataFrame, encoded: pd.DataFrame) -> pd.DataFrame:
    x = df.drop(columns=[ID_COL, TARGET_COL], errors="ignore").reset_index(drop=True).copy()
    return pd.concat([x, encoded.reset_index(drop=True)], axis=1)


def align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_columns]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def apply_calibration(pred: np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    return sigmoid(calibration["scale"] * logit(pred) + calibration["bias"])


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
    target_encoders = artifact["target_encoders"]
    target_encoding_specs = artifact["target_encoding_specs"]
    target_prior = artifact["target_prior"]
    calibration = artifact["calibration"]

    print(f"Load test: {test_path}")
    test = load_test(test_path)
    sub = load_sample_submission(sample_path)
    ids = test[ID_COL].tolist()
    print(f"test={len(test)} submission={len(sub)}")

    print("Build features...")
    encoded = apply_target_encoders(test, target_encoding_specs, target_encoders, target_prior)
    x_test = build_features(test, encoded)
    x_test = align_features(x_test, feature_columns)
    print(f"features={x_test.shape[1]}")

    print("Inference model...")
    preds = model.predict_proba(x_test)[:, 1] if len(x_test) else np.array([])
    preds = apply_calibration(preds, calibration) if len(preds) else preds
    print(f"preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(output_path, sub)
    print(f"Saved: {output_path} rows={len(sub)}")


if __name__ == "__main__":
    main()
