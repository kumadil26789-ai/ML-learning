"""Candidate estimators and their hyper-parameter search spaces."""
from __future__ import annotations

from scipy.stats import loguniform, randint
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


def get_candidates(random_state: int = 42) -> dict:
    return {
        "ridge": (Ridge(), {"alpha": loguniform(1e-2, 1e2)}),
        "random_forest": (
            RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=random_state),
            {"n_estimators": randint(150, 400), "max_depth": [None, 10, 20],
             "min_samples_leaf": randint(1, 8), "max_features": [0.5, 0.8, 1.0]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=random_state),
            {"learning_rate": loguniform(0.02, 0.2), "max_iter": randint(150, 500),
             "max_leaf_nodes": randint(15, 63), "min_samples_leaf": randint(10, 50),
             "l2_regularization": loguniform(1e-4, 1.0)},
        ),
    }


def get_baseline():
    return DummyRegressor(strategy="median")
