"""Loading, row-level cleaning, validation and splitting."""
from __future__ import annotations

import hashlib
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from regml.config import resolve_path
from regml.features import input_columns

log = logging.getLogger(__name__)


def load_raw(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg["data"]["raw_path"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m regml.generate_data` "
                                "or point data.raw_path at your CSV.")
    return pd.read_csv(path)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_rows(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Row-level cleaning that changes the number of rows (so it cannot be a pipeline step)."""
    d = cfg["data"]
    target = d["target"]
    report = {"rows_raw": int(len(df))}

    subset = [c for c in df.columns if c not in d.get("id_columns", [])]
    df = df.drop_duplicates(subset=subset)
    report["duplicates_removed"] = report["rows_raw"] - int(len(df))

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    bad = df[target].isna() | (df[target] < d.get("min_target", 0))
    report["bad_target_removed"] = int(bad.sum())
    df = df.loc[~bad].reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    log.info("Row cleaning: %s", report)
    return df, report


def validate_schema(df: pd.DataFrame, cfg: dict) -> None:
    missing = [c for c in input_columns(cfg) + [cfg["data"]["target"]] if c not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing configured columns: {missing}")


def split(df: pd.DataFrame, cfg: dict):
    d = cfg["data"]
    X = df[input_columns(cfg)]
    y = df[d["target"]]
    return train_test_split(X, y, test_size=d["test_size"], random_state=d["random_state"])
