import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from regml.api import create_app
from regml.cleaning import DataCleaner, Winsorizer
from regml.config import load_config
from regml.data import clean_rows
from regml.generate_data import generate
from regml.predict import Predictor, ValidationError
from regml.train import run_training


class CleaningTests(unittest.TestCase):
    def test_cleaner_handles_dirty_values(self):
        df = pd.DataFrame({"area": ["1,850 sqft", "N/A", -5, None],
                           "cat": [" House ", "UNKNOWN", "condo", np.nan]})
        out = DataCleaner(numeric_cols=["area"], categorical_cols=["cat"],
                          nonnegative_cols=["area"], placeholders=["n/a", "unknown"]).transform(df)
        self.assertEqual(out["area"].iloc[0], 1850.0)
        self.assertTrue(out["area"].iloc[1:].isna().all())          # placeholder, negative, None
        self.assertEqual(list(out["cat"].dropna()), ["house", "condo"])

    def test_cleaner_adds_missing_columns(self):
        out = DataCleaner(numeric_cols=["a"], categorical_cols=["b"]).transform(pd.DataFrame({"x": [1]}))
        self.assertTrue(out[["a", "b"]].isna().all().all())

    def test_winsorizer_clips_but_keeps_nan(self):
        x = np.array([[1.0], [2.0], [3.0], [1000.0], [np.nan]])
        w = Winsorizer(0.0, 0.75).fit(x)
        out = w.transform(x)
        self.assertLess(np.nanmax(out), 1000)
        self.assertTrue(np.isnan(out[-1, 0]))

    def test_row_cleaning(self):
        cfg = load_config()
        df = pd.DataFrame({"id": [1, 2, 3, 4], "a": [1, 1, 2, 3], "price": [10, 10, None, 0]})
        out, rep = clean_rows(df, cfg)
        self.assertEqual(rep["duplicates_removed"], 1)
        self.assertEqual(len(out), 1)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cfg = copy.deepcopy(load_config())
        generate(1200, 1).to_csv(root / "data.csv", index=False)
        cfg["data"]["raw_path"] = str(root / "data.csv")
        cfg["artifacts"] = {"model_dir": str(root / "models"), "report_dir": str(root / "reports")}
        cfg["training"].update(candidates=["ridge", "hist_gradient_boosting"], cv_folds=3, tune=False)
        cls.cfg = cfg
        cls.meta = run_training(cfg)
        cls.predictor = Predictor(cfg["artifacts"]["model_dir"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_beats_baseline_and_is_accurate(self):
        base = self.meta["cv_results"]["baseline_median"]["rmse_mean"]
        self.assertLess(self.meta["test_metrics"]["rmse"], base)
        self.assertGreater(self.meta["test_metrics"]["r2"], 0.7)

    def test_predict_with_nulls_unseen_category_and_text_numbers(self):
        rec = {"area_sqft": "2,000 sqft", "bedrooms": 3, "property_type": "Mansion",
               "neighborhood": "never_seen", "condition": "great", "age_years": None}
        p = self.predictor.predict(rec)
        self.assertEqual(len(p), 1)
        self.assertTrue(np.isfinite(p[0]) and p[0] > 0)

    def test_batch_predict(self):
        recs = [{"area_sqft": 1500 + 100 * i, "bedrooms": 3} for i in range(5)]
        self.assertEqual(len(self.predictor.predict(recs)), 5)

    def test_required_fields_enforced(self):
        with self.assertRaises(ValidationError):
            self.predictor.predict({"bedrooms": 3})
        with self.assertRaises(ValidationError):
            self.predictor.predict([])

    def test_api(self):
        c = create_app(predictor=self.predictor, cfg=self.cfg).test_client()
        self.assertEqual(c.get("/health").status_code, 200)
        self.assertEqual(c.get("/ready").status_code, 200)
        r = c.post("/predict", json={"instances": [{"area_sqft": 1800, "bedrooms": 3}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["count"], 1)
        self.assertEqual(c.post("/predict", json={"bedrooms": 3}).status_code, 400)
        self.assertEqual(c.post("/predict", data="not json").status_code, 400)
        self.assertEqual(c.get("/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
