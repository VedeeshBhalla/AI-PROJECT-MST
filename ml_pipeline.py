"""Reusable estimators; saved pipelines accept raw operational DataFrames."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

import config


class OperationalFeatures(TransformerMixin, BaseEstimator):
    """Select approved columns and learn medians/clipping on fit data only."""

    def _select(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Supply raw inputs as a pandas DataFrame with column names.")
        missing = sorted(set(config.NUMERIC_COLS) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required operational columns: {missing}")
        frame = X.loc[:, config.NUMERIC_COLS].apply(pd.to_numeric, errors="raise")
        if np.isinf(frame.to_numpy(dtype=float)).any():
            raise ValueError("Inputs contain infinite values.")
        return frame

    def fit(self, X, y=None):
        frame = self._select(X)
        if frame.isna().all().any():
            raise ValueError("A training feature is entirely missing.")
        self.imputer_ = SimpleImputer(strategy="median")
        values = self.imputer_.fit_transform(frame)
        q1, q3 = np.quantile(values, [0.25, 0.75], axis=0)
        self.lower_ = q1 - 3 * (q3 - q1)
        self.upper_ = q3 + 3 * (q3 - q1)
        self.feature_names_in_ = np.array(config.NUMERIC_COLS, dtype=object)
        self.n_features_in_ = len(config.NUMERIC_COLS)
        self.output_names_ = self._engineer(values).columns.to_numpy()
        return self

    @staticmethod
    def _engineer(values):
        frame = pd.DataFrame(values, columns=config.NUMERIC_COLS)
        # Raw-unit engineering precedes standardization. No dataset-wide shifts.
        for name, a, b in [
            ("driver_x_fatigue", "driver_behavior_score", "fatigue_monitoring_score"),
            ("port_x_customs", "port_congestion_level", "customs_clearance_time"),
        ]:
            frame[f"feat_{name}"] = frame[a] * frame[b]
        for name, a, b in [
            ("inventory_ratio", "warehouse_inventory_level", "historical_demand"),
            ("loading_ratio", "loading_unloading_time", "handling_equipment_availability"),
        ]:
            frame[f"feat_{name}"] = frame[a] / (1.0 + frame[b].abs())
        for col in ("shipping_costs", "historical_demand", "lead_time_days"):
            frame[f"feat_{col}_log"] = np.sign(frame[col]) * np.log1p(frame[col].abs())
        return frame

    def transform(self, X):
        check_is_fitted(self, "imputer_")
        values = self.imputer_.transform(self._select(X))
        return self._engineer(np.clip(values, self.lower_, self.upper_)).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_names_")
        return self.output_names_.copy()


def make_pipeline(classifier, resample=True):
    steps = [("features", OperationalFeatures()), ("scale", StandardScaler())]
    if resample:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE, k_neighbors=3)))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def load_dataset(path=config.DATA_PATH):
    df = pd.read_csv(path)
    if config.TARGET_COL not in df:
        raise ValueError(f"Missing target column: {config.TARGET_COL}")
    if df[config.TARGET_COL].isna().any():
        raise ValueError("Target labels must not be missing.")
    if set(df[config.TARGET_COL].unique()) != set(config.CLASS_LABELS):
        raise ValueError(f"Expected target classes {config.CLASS_LABELS}.")
    OperationalFeatures()._select(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if df.duplicated(subset=config.NUMERIC_COLS, keep=False).any():
        raise ValueError("Repeated operational inputs found; resolve duplicates/conflicting labels before splitting.")
    return df
