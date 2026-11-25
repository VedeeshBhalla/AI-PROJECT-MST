# Logistics Risk Classification

An educational machine learning experiment that predicts the supplied **High Risk**, **Low Risk**, and **Moderate Risk** labels from operational logistics measurements. It compares Logistic Regression and Random Forest with a majority-class baseline using a reproducible, leakage-safe workflow.

This is an independent student project. No Amazon affiliation, real-world deployment, or observed delivery-delay prediction is claimed.

## Dataset and scope

The included `data/dynamic_supply_chain_logistics_dataset.csv` contains 32,065 rows and 26 columns, with timestamps from January 2021 through August 2024. The target counts are High Risk: 23,944; Moderate Risk: 5,011; Low Risk: 3,110.

Source: [DatasetEngineer, Logistics and supply chain dataset (Kaggle)](https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset), listed as CC0: Public Domain. The public download was verified on 2026-11-25 as a byte-for-byte match to this CSV. See `DATASET.md` for the hash, source definitions and limitations. The original acquisition date, label-generation code and real/synthetic status are unverified. Do not represent the data as Amazon observations or independently authenticated real-world records.

The supplied labels exactly match thresholds of `disruption_likelihood_score` across all rows: below 0.3 is Low Risk, 0.3 to below 0.7 is Moderate Risk, and 0.7 or above is High Risk. A classifier using that score would simply reconstruct the label rule. It is excluded, together with actual `delivery_time_deviation` and the undocumented precomputed scores `delay_probability` and `route_risk_level`. The latter two are excluded conservatively; they have not been established as deterministic encodings of the target.

The source defines `eta_variation_hours` using actual arrival time and `order_fulfillment_status` using whether fulfillment was on time. These outcome inputs are also excluded. The experiment uses a strict allowlist of 18 operational inputs from `config.py`, including GPS, traffic, warehouse stock, weather, shipping costs, driver scores, and cargo sensors. Their measurement times still require verification. This is a classification experiment at an unspecified operational checkpoint; several inputs may not exist before shipment. Risk categories must not be renamed as observed On-Time/Delayed outcomes.

## Run

Use Python 3.11 or newer. The pinned environment was verified with Python 3.14.2 on Windows; other Python/platform combinations have not been tested.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main_pipeline.py
```

On Linux/macOS, activate with `source .venv/bin/activate`. The experiment uses one worker and 100-tree forests to limit resource use. Runtime depends on the machine; progress is printed for each candidate. All outputs are written relative to the repository, regardless of your current working directory.

## Method

1. Validate required inputs and labels, remove exact duplicate records, reject repeated operational inputs that could cross partitions, and reserve a stratified 20% random holdout with seed 42. Development feature summaries use training rows only.
2. Compare three fixed-parameter pipelines with the same shuffled five-fold stratified CV on the remaining 80%. Each fold independently fits median imputation, 3-IQR clipping, seven raw-unit engineered features, and standard scaling. SMOTE with three neighbors runs only inside each training fold for the learned models. The majority baseline uses no SMOTE.
3. Select the highest **mean CV macro F1**, including the baseline as an eligible winner, then refit every candidate on all training rows. Logistic Regression uses `lbfgs`, `C=1`, and 2,000 maximum iterations. Random Forest uses 100 trees, depth 12, and a minimum leaf size of 10. There is no hyperparameter search and no fabricated CV statistic.
4. Evaluate the frozen candidates on the untouched holdout, without changing the selected winner. Report accuracy, macro/weighted F1, balanced accuracy, macro OvR ROC-AUC, per-class precision/recall/F1, confusion matrices, and ROC/precision-recall curves. Probability columns follow the saved classifier's class order.

The seven engineered features are driver score multiplied by fatigue, port congestion multiplied by customs time, two ratios with `1 + abs(denominator)`, and signed `log1p(abs(x))` transforms of costs, demand, and lead time. Engineering happens before scaling and uses no batch-dependent minimum. Saved pipelines include every fitted inference transform.

## Files and outputs

| File | Purpose |
| --- | --- |
| `config.py` | Feature allowlist, excluded columns, split and CV settings |
| `ml_pipeline.py` | Input validation and reusable fitted pipeline components |
| `step1_eda_preprocessing.py` | Raw-data audit, holdout reservation, training-only EDA |
| `step2_feature_engineering.py` | Feature schema preview; no model-training data transformation |
| `step3_train_models.py` | Training-only CV, selection, refitting and saved pipelines |
| `step4_evaluate_report.py` | Holdout metrics, reports and plots |
| `main_pipeline.py` | Runs all four steps in order, stopping on errors |
| `predict.py` | Predicts from a new raw CSV |
| `tests/test_pipeline.py` | Leakage, schema, edge-case and serialization regression checks |

Generated files in `outputs/` are ignored by Git. `results/` contains the text/CSV/JSON report snapshot from the verified final run so reviewers can inspect measured evidence without retraining:

- `outputs/models/`: each complete pipeline, `best_model.pkl`, and its selected name.
- `outputs/reports/`: data audit (including dataset SHA-256), feature schema, training statistics, per-fold and aggregate CV results, actual holdout metrics, per-class reports, feature importance, environment/parameter metadata, and `final_evaluation_report.txt`.
- `outputs/plots/`: training class/feature distributions, metric comparison, confusion matrices, ROC/PR curves, and descriptive Random Forest importance.
- `outputs/data_split.pkl`: raw training/holdout inputs, labels and row identifiers for this run.

Run the full pipeline after changing data, settings or source code. Do not mix artifacts from different runs. Include the generated report and relevant figures in your submission package; they are regenerated locally rather than stored as large binaries in Git. A raw-input prediction example is:

```powershell
python predict.py new_shipments.csv --output outputs/predictions.csv
```

The CSV must contain all 18 operational columns. Target, timestamp, and excluded columns are unnecessary and ignored if present. Missing cells are imputed using training medians; missing columns, nonnumeric values and infinities fail clearly. The output contains one prediction and class probabilities per input row in the original order. Probabilities are **uncalibrated model scores**, especially after SMOTE, rather than established delivery-event probabilities. Only load pickle/joblib artifacts you trust.

## Interpretation and submission limitations

The random split estimates performance on similar rows from the same distribution. It does not establish future forecasting, unseen-vehicle generalization, or robustness to time drift. A prospective application needs a chronological split and entity grouping once identifiers and measurement times are documented.

The holdout was inspected during repository repair before the final source-based exclusion of ETA variation and fulfillment status. The final run keeps selection inside training CV, but its reported holdout scores are exploratory development evidence. A fresh external evaluation is required for an unbiased final-performance claim.

Report the measured baseline and minority-class recall alongside accuracy. A dominant-class prediction alone has substantial accuracy on this dataset. Low performance after removing label proxies is a valid finding; do not restore leaking features or invent accuracy targets to improve the headline. Random Forest impurity importance is descriptive, can be biased, and is not evidence of causality. Confidence histograms are not calibration tests.

See `STEP_BY_STEP_GUIDE.md` for the submission checklist and troubleshooting. Remaining source-authenticity, measurement-timing and institution-specific requirements must be addressed by the student.

## Presentation and project report

- [PowerPoint presentation](submission/deliverables/Logistics_Risk_Classification_Canva_Theme_Final.pptx): 22 slides, styled from the supplied Canva template preview, covering motivation, inputs, training, validation, execution and measured results.
- [Project report](submission/deliverables/Logistics_Risk_Classification_Project_Report.pdf): 20 pages with five chapters covering introduction, literature survey, research gap and objectives, methodology, and results and graphs. The guide and college fields are fillable.

Both deliverables include Vedeesh Bhalla (1024030999) and Udayan Mahalwar (1024031088). They describe the corrected experiment and its weak predictive performance. Local build scripts, previews and temporary files are excluded from Git.
