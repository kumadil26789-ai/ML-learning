# House-Price Regression: production-ready ML project

End-to-end tabular regression with **data cleaning, null handling, encoding, outlier treatment, feature engineering, model selection, tuning, evaluation, artifacts, tests, a REST API and Docker**.
Swap in your own CSV and edit `config/config.yaml`; no code changes needed for most datasets.

## Quick start
```bash
pip install -r requirements.txt
python -m regml.generate_data      # creates a deliberately dirty synthetic CSV (skip if you have real data)
python -m regml.train              # clean -> split -> CV -> tune -> test -> save artifacts
python -m unittest discover -s tests -t . -v
gunicorn --bind 0.0.0.0:8000 "regml.api:create_app()"
```
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d \
 '{"instances":[{"area_sqft":1850,"bedrooms":3,"bathrooms":2,"property_type":"House","condition":"Good","neighborhood":"nbhd_07"}]}'
```
Optional fields may be omitted or null; they are imputed exactly as in training. Only `features.required` are mandatory.

## Layout
```
config/config.yaml      all columns, encodings, thresholds, model & training settings
regml/generate_data.py  dirty synthetic data (missing, placeholders, text numbers, outliers, dupes)
regml/data.py           load, row-level cleaning (dupes, bad targets), schema check, split
regml/cleaning.py       DataCleaner, FeatureEngineer, Winsorizer (sklearn transformers)
regml/features.py       ColumnTransformer + full pipeline builder
regml/models.py         candidate models + search spaces (+ median baseline)
regml/train.py          CV model selection, tuning, test evaluation, artifact saving
regml/predict.py        validated inference wrapper
regml/api.py            Flask service: /health /ready /metadata /predict
tests/                  unit + end-to-end + API tests
```

## What the pipeline does
| Stage | Handling |
|---|---|
| Row cleaning | drop duplicate rows (ignoring ID), drop rows with missing / invalid target |
| Type fixing | `"1,850 sqft"` -> `1850.0`; trim, lowercase, collapse whitespace |
| Placeholders | `n/a`, `?`, `unknown`, `""` ... -> NaN (configurable) |
| Invalid values | negatives in non-negative columns -> NaN |
| Outliers | winsorize to train-set quantiles (0.5%-99.5%) |
| Numeric nulls | median impute **+ missing-indicator columns** + standard scaling |
| Low-cardinality categoricals | impute `"missing"` -> one-hot (rare levels grouped, unseen levels safe) |
| High-cardinality categoricals | impute -> **cross-fitted target encoding** (no target leakage) |
| Ordinal categoricals | most-frequent impute -> ordered encoding (unseen -> -1) |
| Target | trained on `log1p(price)`, predictions returned on original scale |

## Why it is production-ready
- **No leakage.** Every fitted step (quantiles, medians, scalers, encoders) lives inside one sklearn `Pipeline`, fit on training data only. CV re-fits it per fold.
- **No train/serve skew.** The same pipeline object is saved with `joblib` and used by the API.
- **Honest evaluation.** Model choice and tuning use CV on train only; the test set is touched once. A median baseline is reported for context.
- **Robust inference.** Unknown categories, missing fields, text numbers and extra keys are all handled; required fields are validated; clear 400 errors.
- **Traceability.** `models/metadata.json` stores version, data SHA-256, sklearn/Python versions, input contract, CV + test metrics, null rates and output feature names.
- **Ops.** Liveness vs readiness endpoints, request IDs, latency logging, batch size and payload limits, non-root Docker image with healthcheck.
- **Reports.** `reports/metrics.json`, `feature_importance.csv` (permutation, on raw columns), `diagnostics.png`.

## Using your own data
1. Set `data.raw_path`, `data.target`, `data.id_columns`.
2. List columns under `features.*` (numeric / one-hot / target-encode / ordinal order).
3. Set `features.add_engineered: false` (or edit `FeatureEngineer`).
4. `python -m regml.train`, then check `reports/` and `models/metadata.json`.

## Docker
```bash
python -m regml.train && docker build -t house-price-api . && docker run -p 8000:8000 house-price-api
```

## Next steps for a real deployment
Data validation (e.g. Great Expectations / pandera), experiment tracking (MLflow), input and prediction drift monitoring, scheduled retraining with a champion/challenger gate, CI running `make test`, and a model registry instead of a local `models/` folder.
