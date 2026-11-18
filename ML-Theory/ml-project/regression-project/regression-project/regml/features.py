"""Preprocessing + model pipeline construction."""
from __future__ import annotations

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from regml.cleaning import ENGINEERED, DataCleaner, FeatureEngineer, Winsorizer


def categorical_columns(cfg: dict) -> list[str]:
    f = cfg["features"]
    return (list(f.get("categorical_onehot", [])) + list(f.get("categorical_target_encode", []))
            + list(f.get("ordinal", {}).keys()))


def input_columns(cfg: dict) -> list[str]:
    """Raw columns the model consumes (the serving contract)."""
    return list(cfg["features"]["numeric"]) + categorical_columns(cfg)


def _target_encoder(random_state: int) -> TargetEncoder:
    """Deterministic cross-fitting across sklearn versions (API changed in 1.9)."""
    from sklearn.model_selection import KFold
    major_minor = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if major_minor >= (1, 9):
        return TargetEncoder(target_type="continuous", smooth="auto",
                             cv=KFold(5, shuffle=True, random_state=random_state))
    return TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=random_state)


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    f = cfg["features"]
    numeric = list(f["numeric"]) + (ENGINEERED if f.get("add_engineered") else [])
    lo, hi = f.get("winsorize", [0.005, 0.995])
    rs = cfg["data"]["random_state"]

    num_pipe = Pipeline([
        ("winsor", Winsorizer(lo, hi)),
        # median is robust to skew; indicator columns keep "was missing" signal
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    parts = [("num", num_pipe, numeric)]

    if f.get("categorical_onehot"):
        parts.append(("ohe", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                     min_frequency=0.01, sparse_output=False)),
        ]), list(f["categorical_onehot"])))

    if f.get("categorical_target_encode"):
        # TargetEncoder cross-fits internally during fit -> no target leakage
        parts.append(("te", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", _target_encoder(rs)),
        ]), list(f["categorical_target_encode"])))

    if f.get("ordinal"):
        cols = list(f["ordinal"].keys())
        parts.append(("ord", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(categories=[f["ordinal"][c] for c in cols],
                                      handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cols))

    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def param_prefix(cfg: dict) -> str:
    return "regressor__model__" if cfg["training"]["log_target"] else "model__"


def build_model(cfg: dict, estimator):
    """clean -> engineer -> preprocess -> estimator (optionally on log target)."""
    f = cfg["features"]
    steps = [("cleaner", DataCleaner(
        numeric_cols=list(f["numeric"]),
        categorical_cols=categorical_columns(cfg),
        nonnegative_cols=list(f.get("nonnegative", [])),
        placeholders=list(f.get("placeholders", []))))]
    if f.get("add_engineered"):
        steps.append(("engineer", FeatureEngineer()))
    steps += [("preprocess", build_preprocessor(cfg)), ("model", estimator)]
    pipe = Pipeline(steps)
    if cfg["training"]["log_target"]:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                          inverse_func=np.expm1, check_inverse=False)
    return pipe


def inner_pipeline(model) -> Pipeline:
    return getattr(model, "regressor_", model)
