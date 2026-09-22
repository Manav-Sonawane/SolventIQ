import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from common import Xy, build_preprocessor, load_split

train, val, test, numeric_features, categorical_features = load_split()

print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

X_train, y_train = Xy(train, numeric_features, categorical_features)
X_val, y_val = Xy(val, numeric_features, categorical_features)

# Random Forest doesn't need scaled inputs.
preprocessor = build_preprocessor(
    numeric_features, categorical_features, scale_numeric=False
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )),
])

print("\nTraining Random Forest...")
t0 = time.time()
model.fit(X_train, y_train)
print(f"Fit time: {time.time() - t0:.1f}s")

val_proba = model.predict_proba(X_val)[:, 1]
val_pred = (val_proba >= 0.5).astype(int)

roc_auc = roc_auc_score(y_val, val_proba)
pr_auc = average_precision_score(y_val, val_proba)
f1 = f1_score(y_val, val_pred)

print("\nValidation metrics (Random Forest):")
print(f"  ROC-AUC:  {roc_auc:.4f}")
print(f"  PR-AUC:   {pr_auc:.4f}")
print(f"  F1:       {f1:.4f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/random_forest.joblib")
print("\nSaved: models/random_forest.joblib")
