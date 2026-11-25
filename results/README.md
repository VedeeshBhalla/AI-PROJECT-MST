# Verified experiment snapshot

Generated on 2026-11-25 by `python main_pipeline.py`, using the pinned dependencies, Python 3.14.2 and the dataset SHA-256 recorded in `data_audit.json`. The raw split contains 25,652 training rows and 6,413 holdout rows. There are 18 approved raw inputs and 25 total model features. `cv_folds.csv` contains all 15 candidate/fold observations. `run_metadata.json` records parameters, versions and the training-only selection rule. Re-running the experiment writes fresh results to `outputs/`; this directory is a committed snapshot.

Random Forest was selected by five-fold training macro F1. Its holdout macro F1 is 0.3082, versus 0.2850 for the majority baseline, but its holdout accuracy is 0.5444 versus 0.7468 for that baseline. Forest balanced accuracy is 0.3262 and macro OvR ROC-AUC is 0.4989. Together these metrics show poor class discrimination. The small macro F1 gain does not justify a claim of a useful delivery-risk prediction system. The corrected experiment is useful as a study of leakage, imbalance, validation, and dataset limitations.

This holdout was inspected during repository repair before the final source-based removal of outcome features. Treat these scores as exploratory development evidence. A fresh external dataset and documented operational measurement times are required for prospective validation. See `../DATASET.md` for verified provenance and remaining source limitations.

Verification completed:

- Seven regression tests covering excluded inputs, fold-local preprocessing/SMOTE, stable batch-independent transforms, invalid schemas, numerical edge cases, serialization, and duplicate/missing-target handling.
- Full four-step training/evaluation run and a separate seven-row raw-CSV inference smoke check with no labels or excluded columns supplied.
- Disjoint training/holdout row identifiers, expected allowlist, training-only imputer medians, valid model probabilities, CV aggregation/selection consistency, and exact dataset/source hash equality.
- Dependency consistency via `python -m pip check` and syntax checks for all Python files.

Model binaries and raw split artifacts are regenerated locally and excluded from Git. The report files and seven figures in `plots/` are included for review and submission preparation.
