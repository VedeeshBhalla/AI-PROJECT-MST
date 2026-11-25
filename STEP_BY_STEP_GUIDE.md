# Running and presenting the ML experiment

## 1. Prepare the environment

Use Python 3.11+ and install the pinned dependencies in `requirements.txt` in a virtual environment. The tested interpreter is Python 3.14.2 on Windows. Run:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## 2. Audit and reserve the holdout

```powershell
python step1_eda_preprocessing.py
```

The script validates the CSV, checks labels, removes exact duplicates, rejects repeated operational inputs, and reserves a seed-42 stratified 80/20 split before learning any transformation. It writes raw partitions to `outputs/data_split.pkl`, an audit to `outputs/reports/data_audit.json`, and training-only EDA. The audit includes a dataset hash and a count of labels reproduced by the excluded disruption-score threshold rule. Whole-dataset integrity checks are separate from training feature exploration.

## 3. Inspect feature engineering

```powershell
python step2_feature_engineering.py
```

Read `outputs/reports/feature_schema.json`. There are 18 raw inputs plus seven engineered features. A training-only preview documents names. It is never used as a precomputed matrix for cross-validation. Fold-local transforms are fitted independently during training.

## 4. Train and select

```powershell
python step3_train_models.py
```

Each candidate receives the same five stratified folds. Imputation, clipping, scaling and SMOTE are fitted within the training side of each fold. SMOTE never sees validation/holdout rows. The majority-class baseline does not use SMOTE. Macro F1 weights the three classes equally and selects the winner. Accuracy and balanced accuracy are also recorded, with per-fold values and sample standard deviations. Parameters are fixed, with no tuning claim.

The scripts save full pipelines that accept raw named columns. There is no separate manually applied inference scaler. The holdout is not evaluated during selection. A majority baseline may legitimately win if operational measurements provide insufficient useful signal.

## 5. Evaluate once the model choices are frozen

```powershell
python step4_evaluate_report.py
```

Review:

- `cross_validation.csv` and `cv_folds.csv` for training-only validation.
- `model_comparison.csv` for holdout metrics, including the baseline.
- `classification_reports.json` for each class's precision, recall, F1 and support.
- `final_evaluation_report.txt` for the complete report and limitations.
- `run_metadata.json` for actual package versions, parameters and selection criterion.
- The seven PNGs in `outputs/plots/` for EDA, model comparison and evaluation.

This step does not replace the CV-selected winner using holdout results. If you redesign the experiment after inspecting the holdout, disclose that reuse and obtain a fresh final evaluation set before making unbiased final-performance claims.

The checked-in `results/` snapshot is exploratory: this holdout was inspected during repository repair before the final source-based removal of actual-arrival ETA variation and on-time fulfillment status. The pipeline still uses training-only selection. Do not describe this snapshot as an independent external evaluation.

To run all steps in the correct order:

```powershell
python main_pipeline.py
```

## 6. Demonstrate raw-data inference

```powershell
python predict.py new_shipments.csv --output outputs/predictions.csv
```

Provide the 18 columns listed in `config.NUMERIC_COLS`; labels and excluded columns are not needed. Preserve the input row order when joining predictions back to shipments. Results are risk labels, not evidence that a delivery was late. Model probabilities have not been calibrated to real-world prevalence.

## 7. Complete the submission

Before handing this in:

- Cite the verified Kaggle source and CC0 metadata in `DATASET.md`. Explain that source authenticity, original acquisition date, real/synthetic status, measurement timing and label-generation code remain unverified; do not invent these details.
- Explain the exact disruption-score/label relationship and why that score and its derivatives are excluded. Explain the exclusions of undocumented prediction/risk scores, actual delivery outcome, actual-arrival ETA variation and on-time fulfillment status.
- Define the operational prediction checkpoint and confirm that every input is available then. Avoid a pre-shipment claim without this evidence.
- Include actual CV and holdout results, the baseline, per-class recall, confusion matrices and limitations. Use the actual High/Low/Moderate Risk names.
- Explain why a random stratified split was used and why future forecasting would need chronological validation and potentially grouping by shipment/vehicle.
- Add your own analysis and a conclusion that matches the measured results. If the models do not beat the baseline, discuss that finding honestly.
- Include the code, dependency pins, generated report/figures, and any notebook, presentation, originality statement or other material required by your institution. This repository alone cannot establish compliance with an unspecified rubric.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Missing operational columns | Match the CSV schema to the explicit allowlist; do not auto-include every numeric column. |
| Invalid/missing labels | Supply only the three documented target classes and resolve missing targets. |
| Repeated operational inputs | Investigate duplicate records or conflicting targets before splitting. |
| Missing output/model artifacts | Rerun `python main_pipeline.py`; individual steps depend on earlier steps. |
| Dependency import errors | Recreate the virtual environment with the pinned requirements and a supported Python version. |
| Poor scores | Compare with baseline and per-class metrics; investigate source/label quality without adding label proxies. |
| Resource constraints | Reduce forest tree count before the final experiment; retain the validation protocol and document the changed parameters. |

No Amazon affiliation or production readiness is implied by this project.
