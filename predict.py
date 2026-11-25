"""Predict risk categories from raw operational columns in a CSV."""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "predictions.csv")
    parser.add_argument("--model", type=Path, default=config.MODEL_DIR / "best_model.pkl")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Output must differ from the input CSV.")
    frame = pd.read_csv(args.input)
    model = joblib.load(args.model)
    predictions = pd.DataFrame({"predicted_risk": model.predict(frame)})
    for cls, probability in zip(model.classes_, model.predict_proba(frame).T):
        predictions[f"probability_{cls.replace(' ', '_')}"] = probability
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
