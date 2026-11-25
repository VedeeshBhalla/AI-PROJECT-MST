"""Select fixed-parameter candidates by training-only CV macro F1, then refit."""
import json
import platform
import time
from importlib.metadata import version

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

import config
from ml_pipeline import make_pipeline


def main():
    config.ensure_output_dirs()
    split = joblib.load(config.OUTPUT_DIR / "data_split.pkl")
    X, y = split["X_train"], split["y_train"]
    candidates = {
        "Majority_Baseline": make_pipeline(DummyClassifier(strategy="most_frequent"), resample=False),
        "Logistic_Regression": make_pipeline(LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs", random_state=config.RANDOM_STATE)),
        "Random_Forest": make_pipeline(RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=10,
            random_state=config.RANDOM_STATE, n_jobs=1)),
    }
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    summaries, folds = [], []
    for name, pipeline in candidates.items():
        print(f"Cross-validating {name} ({config.CV_FOLDS} folds)...", flush=True)
        start = time.perf_counter()
        scores = cross_validate(pipeline, X, y, cv=cv, n_jobs=1, error_score="raise",
                                scoring={"macro_f1": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"})
        summary = {"Model": name}
        for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
            vals = scores[f"test_{metric}"]
            summary[f"cv_{metric}"] = float(vals.mean())
            summary[f"cv_{metric}_std"] = float(vals.std(ddof=1))
        for i in range(config.CV_FOLDS):
            folds.append({"Model": name, "fold": i + 1, **{
                metric: float(scores[f"test_{metric}"][i])
                for metric in ("macro_f1", "accuracy", "balanced_accuracy")}})
        pipeline.fit(X, y)
        joblib.dump(pipeline, config.MODEL_DIR / f"{name}.pkl")
        summary["training_seconds"] = time.perf_counter() - start
        summaries.append(summary)
        print(f"  CV macro F1: {summary['cv_macro_f1']:.4f}", flush=True)
    results = pd.DataFrame(summaries)
    best_name = results.loc[results.cv_macro_f1.idxmax(), "Model"]
    joblib.dump(candidates[best_name], config.MODEL_DIR / "best_model.pkl")
    joblib.dump(best_name, config.MODEL_DIR / "best_model_name.pkl")
    results.to_csv(config.REPORT_DIR / "cross_validation.csv", index=False)
    pd.DataFrame(folds).to_csv(config.REPORT_DIR / "cv_folds.csv", index=False)
    metadata = {
        "selected_model": best_name, "selection_metric": "training CV macro F1",
        "cv_folds": config.CV_FOLDS, "random_state": config.RANDOM_STATE,
        "hyperparameter_search": False, "python": platform.python_version(),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "imbalanced-learn", "joblib", "scipy", "matplotlib")},
        "parameters": {name: pipe.named_steps["classifier"].get_params() for name, pipe in candidates.items()},
    }
    (config.REPORT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected {best_name} without evaluating the test set.", flush=True)


if __name__ == "__main__":
    main()
