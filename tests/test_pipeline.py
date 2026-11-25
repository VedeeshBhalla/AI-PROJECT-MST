"""Regression tests for leakage boundaries and deployable preprocessing."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

import config
from ml_pipeline import OperationalFeatures, load_dataset, make_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.X = pd.DataFrame(rng.normal(size=(90, len(config.NUMERIC_COLS))), columns=config.NUMERIC_COLS)
        self.y = np.array([config.CLASS_LABELS[0]] * 45 + [config.CLASS_LABELS[1]] * 30 + [config.CLASS_LABELS[2]] * 15)

    def test_excluded_columns_and_order_cannot_change_features(self):
        fitted = OperationalFeatures().fit(self.X)
        extra = self.X.copy()
        for col in config.EXCLUDED_COLS + [config.TARGET_COL, "label"]:
            extra[col] = np.arange(len(extra))
        np.testing.assert_array_equal(fitted.transform(self.X), fitted.transform(extra[extra.columns[::-1]]))
        self.assertEqual(len(fitted.get_feature_names_out()), 25)

    def test_transform_does_not_fit_on_holdout_or_batch(self):
        self.X.iloc[0, 0] = np.nan
        fitted = OperationalFeatures().fit(self.X)
        medians, lower, upper = fitted.imputer_.statistics_.copy(), fitted.lower_.copy(), fitted.upper_.copy()
        test = self.X.iloc[:3].copy()
        test.iloc[1, :] = 1e9
        alone = fitted.transform(test.iloc[:1])
        batch = fitted.transform(test)
        np.testing.assert_array_equal(alone, batch[:1])
        np.testing.assert_array_equal(medians, fitted.imputer_.statistics_)
        np.testing.assert_array_equal(lower, fitted.lower_)
        np.testing.assert_array_equal(upper, fitted.upper_)
        self.assertTrue(np.isfinite(batch).all())

    def test_zero_negative_and_missing_values_remain_finite(self):
        fitted = OperationalFeatures().fit(self.X)
        test = self.X.iloc[:4].copy()
        test.iloc[0] = 0
        test.iloc[1] = -1
        test.iloc[2] = np.nan
        self.assertTrue(np.isfinite(fitted.transform(test)).all())

    def test_schema_and_invalid_values_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            OperationalFeatures().fit(self.X.drop(columns=config.NUMERIC_COLS[0]))
        invalid = self.X.copy()
        invalid.iloc[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            OperationalFeatures().fit(invalid)
        invalid.iloc[:, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "entirely missing"):
            OperationalFeatures().fit(invalid)

    def test_cv_fits_preprocessing_on_each_training_fold(self):
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        pipeline = make_pipeline(LogisticRegression(max_iter=300))
        scores = cross_validate(pipeline, self.X, self.y, cv=cv, return_estimator=True, error_score="raise")
        for (train, validation), estimator in zip(cv.split(self.X, self.y), scores["estimator"]):
            features = estimator.named_steps["features"]
            np.testing.assert_allclose(features.imputer_.statistics_, self.X.iloc[train].median().values)
            # SMOTE balances only the training fold, not the validation fold.
            counts = pd.Series(self.y[train]).value_counts()
            self.assertEqual(sum(estimator.named_steps["smote"].sampling_strategy_.values()),
                             int(counts.max() * len(counts) - len(train)))
            self.assertEqual(len(estimator.predict(self.X.iloc[validation])), len(validation))

    def test_saved_pipeline_predicts_raw_rows_without_target(self):
        model = make_pipeline(LogisticRegression(max_iter=300)).fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            joblib.dump(model, path)
            restored = joblib.load(path)
            np.testing.assert_array_equal(model.predict(self.X[:5]), restored.predict(self.X[:5]))
            np.testing.assert_allclose(restored.predict_proba(self.X[:5]).sum(axis=1), 1)

    def test_dataset_rejects_duplicate_inputs_and_missing_targets(self):
        data = self.X.copy()
        data[config.TARGET_COL] = self.y
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            duplicate = pd.concat([data, data.iloc[:1]], ignore_index=True)
            duplicate["timestamp"] = np.arange(len(duplicate))
            duplicate.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Repeated operational"):
                load_dataset(path)
            data.loc[0, config.TARGET_COL] = np.nan
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "must not be missing"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
