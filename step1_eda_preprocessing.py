"""Audit raw data and reserve a stratified holdout before fitting transforms."""
import hashlib
import json
import joblib
from sklearn.model_selection import train_test_split

import config
from ml_pipeline import load_dataset


def main():
    config.ensure_output_dirs()
    df = load_dataset()
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET_COL],
    )
    if train[config.TARGET_COL].value_counts().min() < config.CV_FOLDS * 4:
        raise ValueError("Insufficient training examples per class for CV with SMOTE.")
    joblib.dump({
        "X_train": train[config.NUMERIC_COLS], "y_train": train[config.TARGET_COL],
        "X_test": test[config.NUMERIC_COLS], "y_test": test[config.TARGET_COL],
        "train_rows": train.index.to_list(), "test_rows": test.index.to_list(),
    }, config.OUTPUT_DIR / "data_split.pkl")
    audit = {
        "dataset_sha256": hashlib.sha256(config.DATA_PATH.read_bytes()).hexdigest(),
        "rows_after_deduplication": len(df), "columns": df.columns.to_list(),
        "class_counts": df[config.TARGET_COL].value_counts().to_dict(),
        "missing_counts": df.isna().sum().to_dict(),
        "operational_features": config.NUMERIC_COLS, "excluded_features": config.EXCLUDED_COLS,
        "train_rows": len(train), "test_rows": len(test),
        "split": "stratified random holdout; evaluates same-distribution classification, not future forecasting",
        "test_fraction": config.TEST_SIZE,
        "timestamp_range": [str(df.timestamp.min()), str(df.timestamp.max())] if "timestamp" in df else None,
        "dataset_source": "https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset",
        "source_creator": "DatasetEngineer",
        "source_license": "CC0: Public Domain (Kaggle metadata)",
        "source_verified_date": "2026-11-25",
        "verified_source_sha256": "3b90b86fb658823687e461f6444dc10bbde325ea1c37733da92b0dad7071b10e",
    }
    if "disruption_likelihood_score" in df:
        rule = df.disruption_likelihood_score.map(
            lambda v: "Low Risk" if v < 0.3 else "Moderate Risk" if v < 0.7 else "High Risk")
        audit["disruption_rule_matching_rows"] = int((rule == df[config.TARGET_COL]).sum())
    (config.REPORT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    # Development EDA uses training observations only.
    train[config.NUMERIC_COLS].describe().to_csv(config.REPORT_DIR / "training_summary.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train[config.TARGET_COL].value_counts().plot.bar(title="Training class distribution")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "01_training_distribution.png", dpi=150)
    plt.close()
    train[config.NUMERIC_COLS].hist(figsize=(16, 14), bins=30)
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "02_training_features.png", dpi=150)
    plt.close()
    print(f"Reserved {len(test)} holdout rows; {len(train)} training rows. No preprocessing fitted.", flush=True)


if __name__ == "__main__":
    main()
