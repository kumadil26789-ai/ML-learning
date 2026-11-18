"""Stateless / train-fitted cleaning transformers (sklearn-compatible).

They live *inside* the pipeline so exactly the same cleaning is applied at
training time and at inference time (no train/serve skew).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

DEFAULT_PLACEHOLDERS = ("", "n/a", "na", "nan", "null", "?", "-", "unknown", "missing")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _to_number(v, placeholders: set) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().lower()
    if s in placeholders:
        return np.nan
    m = _NUM_RE.search(s.replace(",", ""))
    return float(m.group()) if m else np.nan


def _to_clean_str(v, placeholders: set):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = re.sub(r"\s+", " ", str(v).strip().lower())
    return np.nan if s in placeholders else s


class DataCleaner(BaseEstimator, TransformerMixin):
    """Normalise raw input into clean dtypes.

    - Guarantees every expected column exists (missing ones become NaN)
    - Categoricals: trim, collapse whitespace, lowercase, placeholders -> NaN
    - Numerics: "1,850 sqft" -> 1850.0, placeholders -> NaN
    - Columns in `nonnegative_cols`: values < 0 -> NaN
    """

    def __init__(self, numeric_cols=(), categorical_cols=(), nonnegative_cols=(),
                 placeholders=DEFAULT_PLACEHOLDERS):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.nonnegative_cols = nonnegative_cols
        self.placeholders = placeholders

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataCleaner expects a pandas DataFrame")
        X = X.copy()
        ph = {str(p).strip().lower() for p in self.placeholders}

        for c in self.categorical_cols:
            if c not in X.columns:
                X[c] = np.nan
            X[c] = pd.Series(
                [_to_clean_str(v, ph) for v in X[c].astype(object)], index=X.index, dtype=object
            )
        for c in self.numeric_cols:
            if c not in X.columns:
                X[c] = np.nan
            s = X[c]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                X[c] = s.astype(float)
            else:
                X[c] = pd.Series([_to_number(v, ph) for v in s], index=X.index, dtype=float)
            if c in set(self.nonnegative_cols):
                X.loc[X[c] < 0, c] = np.nan
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain ratios (NaN-safe). Dataset-specific: edit for your own data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["total_rooms"] = X["bedrooms"] + X["bathrooms"]
        X["area_per_room"] = X["area_sqft"] / X["total_rooms"].replace(0, np.nan)
        X["lot_to_area"] = X["lot_size"] / X["area_sqft"].replace(0, np.nan)
        return X


ENGINEERED = ["total_rooms", "area_per_room", "lot_to_area"]


class Winsorizer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to quantile bounds learned on the TRAINING data only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        arr = np.asarray(X, dtype=float)
        self.n_features_in_ = arr.shape[1]
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)  # NaN preserved
