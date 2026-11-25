"""Document training feature schema; fitted transformations stay inside CV."""
import json
import joblib
import config
from ml_pipeline import OperationalFeatures


def main():
    config.ensure_output_dirs()
    split = joblib.load(config.OUTPUT_DIR / "data_split.pkl")
    # Preview is never used as model input. CV fits its own transformer.
    preview = OperationalFeatures().fit(split["X_train"])
    schema = {
        "raw_features": config.NUMERIC_COLS,
        "model_features": preview.get_feature_names_out().tolist(),
        "method": "Training-fold median imputation, 3-IQR clipping, raw-unit interactions/ratios/signed logs, standard scaling",
    }
    (config.REPORT_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Feature schema: {len(schema['raw_features'])} raw -> {len(schema['model_features'])} model features.", flush=True)


if __name__ == "__main__":
    main()
