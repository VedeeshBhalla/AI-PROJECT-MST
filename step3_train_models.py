"""
step3_train_models.py
======================
STEP 3 — Model Training

PERFORMANCE OPTIMISED:
  • Random Forest  : n_estimators=100, no CV search, single fit only
  • Logistic Reg   : no CV loop, single fit only
  • SMOTE          : reduced to minority-only sampling
  • n_jobs=1       : prevents CPU thread overload / crashes
  • No RandomizedSearchCV — fixed params only (saves 20× refit cycles)

Target accuracy ~88–91% (RF), ~78–83% (LR)
"""

import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import config

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

print("\n" + "=" * 60)
print("  STEP 3 — MODEL TRAINING  (Optimised for low RAM/CPU)")
print("  Models: Logistic Regression  |  Random Forest")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df           = pd.read_csv(os.path.join(config.OUTPUT_DIR, "engineered_data.csv"))
all_features = joblib.load(os.path.join(config.MODEL_DIR, "all_feature_cols.pkl"))
all_features = [f for f in all_features if f in df.columns]

X = df[all_features].fillna(0).values
y = df["label"].values

print(f"\n✔ Feature matrix : {X.shape}")
unique, counts = np.unique(y, return_counts=True)
print(f"  Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = config.TEST_SIZE,
    random_state = config.RANDOM_STATE,
    stratify     = y,
)
print(f"\n  Train : {X_train.shape[0]:,} samples")
print(f"  Test  : {X_test.shape[0]:,} samples")

joblib.dump((X_test, y_test),
            os.path.join(config.MODEL_DIR, "test_split.pkl"))
joblib.dump(all_features,
            os.path.join(config.MODEL_DIR, "final_feature_cols.pkl"))

# ─────────────────────────────────────────────────────────────────────────────
# SMOTE — minority classes only, keeps dataset small
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Applying SMOTE (minority only — keeps training set small)...")
smote = SMOTE(
    sampling_strategy = "not majority",  # only upsample minority classes
    random_state      = config.RANDOM_STATE,
    k_neighbors       = 3,               # reduced from 5 → faster, less RAM
)
X_res, y_res = smote.fit_resample(X_train, y_train)
unique_r, counts_r = np.unique(y_res, return_counts=True)
print(f"  After SMOTE : {X_res.shape[0]:,} training samples")
print(f"  Balanced    : {dict(zip(unique_r.tolist(), counts_r.tolist()))}")

results    = {}
best_acc   = 0.0
best_name  = None
best_model = None

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — LOGISTIC REGRESSION
# Single fit only — no cross-validation loop → very fast, low RAM
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  ── Training: Logistic Regression ──────────────────────────")
print("     Single fit, no CV loop  |  C=0.05  |  ~78–83% expected")
t0 = time.time()

lr_clf = LogisticRegression(
    max_iter     = 1000,       # reduced from 3000 → faster convergence
    C            = 0.05,       # strong regularisation
    solver       = "saga",     # saga handles large datasets better than lbfgs
    n_jobs       = 1,          # single thread → prevents CPU overload
    random_state = config.RANDOM_STATE,
)

lr_clf.fit(X_res, y_res)
lr_test_acc = accuracy_score(y_test, lr_clf.predict(X_test))
lr_elapsed  = time.time() - t0

print(f"\n    Test Accuracy: {lr_test_acc:.4f}  ({lr_test_acc*100:.2f}%)")
print(f"    Train Time   : {lr_elapsed:.1f}s")

joblib.dump(lr_clf, os.path.join(config.MODEL_DIR, "Logistic_Regression.pkl"))
results["Logistic_Regression"] = {
    "cv_accuracy":   lr_test_acc,   # using test acc as proxy (no CV)
    "cv_std":        0.0,
    "test_accuracy": lr_test_acc,
    "train_time_s":  lr_elapsed,
}

if lr_test_acc > best_acc:
    best_acc   = lr_test_acc
    best_name  = "Logistic_Regression"
    best_model = lr_clf

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — RANDOM FOREST
# Fixed lightweight params — no RandomizedSearchCV → runs ONE time only
#   n_estimators=100   → enough trees, won't hog RAM like 500
#   max_depth=10       → shallow trees, fast to build
#   n_jobs=1           → single thread → NO CPU crash
#   min_samples_leaf=20→ larger leaves → less RAM per tree
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  ── Training: Random Forest ────────────────────────────────")
print("     100 trees  |  max_depth=10  |  n_jobs=1  |  ~88–91% expected")
print("     (Single fit — no CV search → runs once, no crashes)")
t0 = time.time()

rf_clf = RandomForestClassifier(
    n_estimators      = 100,   # was 500 → 5× less RAM & time
    max_depth         = 10,    # shallow → fast
    min_samples_split = 40,
    min_samples_leaf  = 20,
    max_features      = "sqrt",
    class_weight      = "balanced",
    random_state      = config.RANDOM_STATE,
    n_jobs            = 1,     # CRITICAL: single thread → no crash
)

rf_clf.fit(X_res, y_res)
rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test))
rf_elapsed  = time.time() - t0

print(f"\n    Test Accuracy: {rf_test_acc:.4f}  ({rf_test_acc*100:.2f}%)")
print(f"    Train Time   : {rf_elapsed:.1f}s")

joblib.dump(rf_clf, os.path.join(config.MODEL_DIR, "Random_Forest.pkl"))
results["Random_Forest"] = {
    "cv_accuracy":   rf_test_acc,
    "cv_std":        0.0,
    "test_accuracy": rf_test_acc,
    "train_time_s":  rf_elapsed,
}

if rf_test_acc > best_acc:
    best_acc   = rf_test_acc
    best_name  = "Random_Forest"
    best_model = rf_clf

# ─────────────────────────────────────────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(best_model, os.path.join(config.MODEL_DIR, "best_model.pkl"))
joblib.dump(best_name,  os.path.join(config.MODEL_DIR, "best_model_name.pkl"))

print(f"\n{'─'*55}")
print(f"  🏆 Best model   : {best_name}")
print(f"  🎯 Test Accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%)")
print(f"{'─'*55}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
res_df = (
    pd.DataFrame(results)
    .T.reset_index()
    .rename(columns={"index": "Model"})
)
res_df.to_csv(os.path.join(config.REPORT_DIR, "model_comparison.csv"),
              index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON CHART
# ─────────────────────────────────────────────────────────────────────────────
model_names = list(results.keys())
test_accs   = [results[m]["test_accuracy"] for m in model_names]
colors      = ["#27ae60" if m == best_name else "#3498db" for m in model_names]
bar_labels  = ["Logistic\nRegression", "Random\nForest"]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(bar_labels, test_accs, color=colors,
              edgecolor="black", width=0.45)
ax.set_ylim(max(0, min(test_accs) - 0.15), 1.03)
ax.axhline(0.90, color="red", linestyle="--",
           linewidth=1.8, label="90% Target")
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("Model Comparison — Test Accuracy\n(Green = Best Model)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)

for bar, acc in zip(bars, test_accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{acc*100:.2f}%",
        ha="center", va="bottom", fontsize=13, fontweight="bold",
    )

plt.tight_layout()
plt.savefig(os.path.join(config.PLOT_DIR, "05_model_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved → 05_model_comparison.png")

print(f"""
  ┌──────────────────────────┬──────────────────┬──────────────────┐
  │ Metric                   │ Logistic Reg.    │ Random Forest    │
  ├──────────────────────────┼──────────────────┼──────────────────┤
  │ Test Accuracy            │ {lr_test_acc:.4f}           │ {rf_test_acc:.4f}           │
  │ Train Time               │ {lr_elapsed:5.1f}s           │ {rf_elapsed:5.1f}s           │
  │ Best Model?              │ {"✅ YES" if best_name=="Logistic_Regression" else "   NO "}            │ {"✅ YES" if best_name=="Random_Forest" else "   NO "}            │
  └──────────────────────────┴──────────────────┴──────────────────┘
""")
print("\n✔ STEP 3 COMPLETE\n")
