import numpy as np
import joblib
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from common import Xy, load_split

train, val, test, numeric_features, categorical_features = load_split()
X_test, y_test = Xy(test, numeric_features, categorical_features)

print(f"Test set: {len(test):,} rows  (issue_d range: "
      f"{test['issue_d'].min()}..{test['issue_d'].max()}, "
      f"default rate: {y_test.mean():.4f})")

lr_model = joblib.load("models/logistic_regression.joblib")
rf_model = joblib.load("models/random_forest.joblib")
meta_model = joblib.load("models/meta_model.joblib")

lr_proba = lr_model.predict_proba(X_test)[:, 1]
rf_proba = rf_model.predict_proba(X_test)[:, 1]
stacked_proba = meta_model.predict_proba(np.column_stack([lr_proba, rf_proba]))[:, 1]

results = {
    "Logistic Regression": lr_proba,
    "Random Forest": rf_proba,
    "Stacked Ensemble": stacked_proba,
}

print("\n=== Test set results ===\n")
print(f"{'Model':<22}{'ROC-AUC':>10}{'PR-AUC':>10}{'F1':>10}")
for name, proba in results.items():
    pred = (proba >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    print(f"{name:<22}{roc_auc:>10.4f}{pr_auc:>10.4f}{f1:>10.4f}")

print("\n=== Confusion matrix (Stacked Ensemble, threshold=0.5) ===")
pred = (stacked_proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f"TN={tn:,}  FP={fp:,}")
print(f"FN={fn:,}  TP={tp:,}")
print(f"Recall (defaults caught): {tp / (tp + fn):.4f}")
print(f"Precision (of flagged, actually default): {tp / (tp + fp):.4f}")
