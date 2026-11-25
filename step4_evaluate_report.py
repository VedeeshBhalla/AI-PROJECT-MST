"""Evaluate frozen models on the holdout; never change selection here."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, ConfusionMatrixDisplay,
                             f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize

import config


def main():
    config.ensure_output_dirs()
    split = joblib.load(config.OUTPUT_DIR / "data_split.pkl")
    X, y = split["X_test"], split["y_test"]
    best_name = joblib.load(config.MODEL_DIR / "best_model_name.pkl")
    cv = pd.read_csv(config.REPORT_DIR / "cross_validation.csv")
    summaries, details = [], {}
    fig, axes = plt.subplots(1, len(cv), figsize=(18, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    fig_pr, ax_pr = plt.subplots(figsize=(10, 7))
    for ax, name in zip(np.atleast_1d(axes), cv.Model):
        model = joblib.load(config.MODEL_DIR / f"{name}.pkl")
        pred, prob = model.predict(X), model.predict_proba(X)
        classes = model.classes_.tolist()
        binary = label_binarize(y, classes=classes)
        summaries.append({
            "Model": name, "selected_by_cv": name == best_name,
            "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "roc_auc_macro_ovr": roc_auc_score(y, prob, labels=classes, multi_class="ovr", average="macro"),
        })
        details[name] = classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)
        ConfusionMatrixDisplay.from_predictions(y, pred, labels=classes, ax=ax, colorbar=False,
                                               xticks_rotation=25)
        ax.set_title(name.replace("_", " "))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(binary[:, i], prob[:, i])
            ax_roc.plot(fpr, tpr, label=f"{name}: {cls}")
            precision, recall, _ = precision_recall_curve(binary[:, i], prob[:, i])
            ap = average_precision_score(binary[:, i], prob[:, i])
            ax_pr.plot(recall, precision, label=f"{name}: {cls} (AP={ap:.3f})")
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / "04_confusion_matrices.png", dpi=150)
    plt.close(fig)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    for chart, ax, path, xlabel, ylabel in [
        (fig_roc, ax_roc, "05_roc_curves.png", "False positive rate", "True positive rate"),
        (fig_pr, ax_pr, "06_precision_recall.png", "Recall", "Precision"),
    ]:
        ax.set(xlabel=xlabel, ylabel=ylabel, title="Holdout evaluation")
        ax.legend(fontsize=7)
        chart.tight_layout()
        chart.savefig(config.PLOT_DIR / path, dpi=150)
        plt.close(chart)
    results = pd.DataFrame(summaries)
    results.to_csv(config.REPORT_DIR / "model_comparison.csv", index=False)
    (config.REPORT_DIR / "classification_reports.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cv.plot.bar(x="Model", y="cv_macro_f1", yerr="cv_macro_f1_std", ax=axes[0], legend=False,
                title="Training CV macro F1 (sample SD)", rot=20)
    results.plot.bar(x="Model", y=["accuracy", "macro_f1", "balanced_accuracy"], ax=axes[1],
                     title="Holdout metrics", rot=20)
    for ax in axes:
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(config.PLOT_DIR / "03_model_comparison.png", dpi=150)
    plt.close(fig)

    # Model-specific descriptive importance; not a causal explanation.
    rf = joblib.load(config.MODEL_DIR / "Random_Forest.pkl")
    importance = pd.Series(rf.named_steps["classifier"].feature_importances_,
                           index=rf.named_steps["features"].get_feature_names_out()).sort_values()
    importance.rename("impurity_importance").to_csv(config.REPORT_DIR / "rf_feature_importance.csv")
    importance.tail(20).plot.barh(figsize=(10, 7), title="Random Forest impurity importance (not causal)")
    plt.tight_layout()
    plt.savefig(config.PLOT_DIR / "07_feature_importance.png", dpi=150)
    plt.close()

    selected = results.set_index("Model").loc[best_name]
    baseline = results.set_index("Model").loc["Majority_Baseline"]
    report = [
        "LOGISTICS RISK CLASSIFICATION - EDUCATIONAL EXPERIMENT",
        f"Selected model: {best_name} (training CV macro F1; no test-based selection)",
        f"Holdout rows: {len(y)}", "", "TRAINING CROSS-VALIDATION", cv.to_string(index=False),
        "", "HOLDOUT METRICS", results.to_string(index=False),
        "", "INTERPRETATION",
        f"Selected-model holdout macro F1: {selected.macro_f1:.4f}; majority baseline: {baseline.macro_f1:.4f}.",
        f"Selected-model holdout accuracy: {selected.accuracy:.4f}; majority baseline: {baseline.accuracy:.4f}.",
        f"Selected-model balanced accuracy: {selected.balanced_accuracy:.4f}; macro OvR ROC-AUC: {selected.roc_auc_macro_ovr:.4f}.",
        "Assess these metrics together; a macro F1 improvement alone does not establish useful operational discrimination.",
        "", "PER-CLASS REPORTS", json.dumps(details, indent=2), "", "LIMITATIONS",
        "Source: DatasetEngineer, Kaggle logistics-and-supply-chain-dataset, CC0 metadata; see DATASET.md.",
        "Label-generation code, independent real-world authenticity and operational feature timing are unverified.",
        "Labels exactly match disruption-score thresholds in the supplied data; that score is excluded.",
        "Other undocumented risk scores and actual delivery deviation are excluded conservatively.",
        "Actual-arrival ETA variation and on-time fulfillment status are excluded per source definitions.",
        "Random holdout estimates same-distribution classification, not future deployment performance.",
        "The same holdout was inspected during repository repair; results are exploratory, not a fresh external evaluation.",
        "SMOTE creates synthetic observations; probabilities are not calibrated to deployment prevalence.",
        "The target describes risk categories, not observed delivery-delay events.",
        "If learned models fail to beat the majority baseline, do not claim useful predictive performance.",
        "No Amazon affiliation or production readiness is claimed.",
    ]
    (config.REPORT_DIR / "final_evaluation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Evaluation saved. Selected model remains {best_name}.", flush=True)


if __name__ == "__main__":
    main()
