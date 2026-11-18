"""Training entry point:  python -m regml.train"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate

from regml import __version__
from regml.config import load_config, resolve_path, setup_logging
from regml.data import clean_rows, file_sha256, load_raw, split, validate_schema
from regml.evaluate import compute_metrics, save_plots
from regml.features import (build_model, categorical_columns, inner_pipeline,
                            input_columns, param_prefix)
from regml.models import get_baseline, get_candidates

log = logging.getLogger("regml.train")
SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _cv_summary(cfg, name, estimator, X, y, cv):
    t0 = time.time()
    res = cross_validate(build_model(cfg, estimator), X, y, cv=cv, scoring=SCORING,
                         n_jobs=cfg["training"]["n_jobs"])
    out = {
        "rmse_mean": float(-res["test_rmse"].mean()), "rmse_std": float(res["test_rmse"].std()),
        "mae_mean": float(-res["test_mae"].mean()), "r2_mean": float(res["test_r2"].mean()),
        "seconds": round(time.time() - t0, 1),
    }
    log.info("CV %-24s RMSE=%.0f ± %.0f | MAE=%.0f | R2=%.3f", name,
             out["rmse_mean"], out["rmse_std"], out["mae_mean"], out["r2_mean"])
    return out


def run_training(cfg: dict) -> dict:
    t = cfg["training"]
    out_dir = resolve_path(cfg["artifacts"]["model_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = resolve_path(cfg["artifacts"]["report_dir"]); rep_dir.mkdir(parents=True, exist_ok=True)

    # 1. data ---------------------------------------------------------------
    raw = load_raw(cfg)
    validate_schema(raw, cfg)
    df, row_report = clean_rows(raw, cfg)
    X_train, X_test, y_train, y_test = split(df, cfg)
    log.info("Train=%d  Test=%d", len(X_train), len(X_test))
    null_rates = X_train.isna().mean().round(4).to_dict()

    # 2. model selection via CV on TRAIN only --------------------------------
    cv = KFold(t["cv_folds"], shuffle=True, random_state=t["random_state"])
    cv_results = {"baseline_median": _cv_summary(cfg, "baseline_median", get_baseline(), X_train, y_train, cv)}
    candidates = get_candidates(t["random_state"])
    for name in t["candidates"]:
        cv_results[name] = _cv_summary(cfg, name, candidates[name][0], X_train, y_train, cv)
    best_name = min(t["candidates"], key=lambda n: cv_results[n]["rmse_mean"])
    log.info("Best candidate by CV RMSE: %s", best_name)

    # 3. tune + refit best ----------------------------------------------------
    estimator, space = candidates[best_name]
    best_params = {}
    if t["tune"]:
        prefix = param_prefix(cfg)
        search = RandomizedSearchCV(
            build_model(cfg, estimator), {prefix + k: v for k, v in space.items()},
            n_iter=t["tune_iterations"], cv=cv, scoring=SCORING["rmse"],
            n_jobs=t["n_jobs"], random_state=t["random_state"], refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = {k.replace(prefix, ""): v for k, v in search.best_params_.items()}
        log.info("Tuned CV RMSE=%.0f params=%s", -search.best_score_, best_params)
    else:
        model = build_model(cfg, estimator).fit(X_train, y_train)

    # 4. honest evaluation on untouched test set ------------------------------
    pred = model.predict(X_test)
    metrics = compute_metrics(y_test, pred)
    log.info("TEST  RMSE=%.0f | MAE=%.0f | MAPE=%.1f%% | R2=%.3f",
             metrics["rmse"], metrics["mae"], metrics["mape"] * 100, metrics["r2"])
    save_plots(y_test, pred, rep_dir)

    imp = permutation_importance(model, X_test, y_test, scoring=SCORING["rmse"], n_repeats=5,
                                 random_state=t["random_state"], n_jobs=t["n_jobs"])
    importance = (pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("importance", ascending=False))
    importance.to_csv(rep_dir / "feature_importance.csv", index=False)

    # 5. persist artifacts ----------------------------------------------------
    version = f"{best_name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    metadata = _jsonable({
        "model_version": version, "model_name": best_name, "best_params": best_params,
        "trained_at": datetime.now(timezone.utc).isoformat(), "package_version": __version__,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "target": cfg["data"]["target"], "log_target": t["log_target"],
        "input_columns": input_columns(cfg), "required_fields": cfg["features"]["required"],
        "categorical_columns": categorical_columns(cfg),
        "output_feature_names": list(inner_pipeline(model).named_steps["preprocess"].get_feature_names_out()),
        "data_sha256": file_sha256(resolve_path(cfg["data"]["raw_path"])),
        "row_report": row_report, "train_rows": len(X_train), "test_rows": len(X_test),
        "train_null_rates": null_rates, "cv_results": cv_results, "test_metrics": metrics,
    })
    joblib.dump(model, out_dir / "model.joblib", compress=3)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (rep_dir / "metrics.json").write_text(json.dumps(
        {"cv": _jsonable(cv_results), "test": metrics}, indent=2))
    log.info("Saved model -> %s", out_dir / "model.joblib")
    return metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
