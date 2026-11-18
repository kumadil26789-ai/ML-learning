"""Inference wrapper with input validation. Depends only on saved artifacts."""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from regml.config import load_config, resolve_path

log = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised for bad client input (maps to HTTP 400)."""


class Predictor:
    def __init__(self, model_dir=None):
        d = resolve_path(model_dir or load_config()["artifacts"]["model_dir"])
        self.model = joblib.load(d / "model.joblib")
        self.metadata = json.loads((d / "metadata.json").read_text())
        self.input_columns = self.metadata["input_columns"]
        self.required = self.metadata.get("required_fields", [])

    def _to_frame(self, records) -> pd.DataFrame:
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list) or not records:
            raise ValidationError("Provide a non-empty JSON object or list of objects.")
        for i, r in enumerate(records):
            if not isinstance(r, dict):
                raise ValidationError(f"Record {i} must be an object.")
            absent = [c for c in self.required if r.get(c) in (None, "")]
            if absent:
                raise ValidationError(f"Record {i} is missing required fields: {absent}")
        # unknown keys are ignored; absent optional fields become NaN and are imputed
        return pd.DataFrame(records).reindex(columns=self.input_columns)

    def predict(self, records) -> list[float]:
        preds = self.model.predict(self._to_frame(records))
        return [round(float(p), 2) for p in np.asarray(preds)]
