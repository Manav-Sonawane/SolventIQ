import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from common import Xy, build_preprocessor, load_split

train, val, test, numeric_features, categorical_features = load_split()

X_train, y_train = Xy(train, numeric_features, categorical_features)
X_val, y_val = Xy(val, numeric_features, categorical_features)

print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

# --------------------------------------------------
# Step 17: out-of-fold base-model predictions on TRAIN
# --------------------------------------------------
# The meta-model must never see a base model's prediction
# on a row that base model was trained on - that would let
# the meta-model learn to trust an overfit base prediction.
# cross_val_predict trains a fresh copy of each base pipeline
# per fold and predicts only on that fold's held-out rows.
#
# Reduced estimators/folds vs. the standalone RF (11_random_forest.py)
# purely for runtime - this is 3x the fits of a single RF train.

cv = KFold(n_splits=3, shuffle=True, random_state=42)

lr_pipeline = Pipeline([
    ("preprocess", build_preprocessor(numeric_features, categorical_features, scale_numeric=True)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

rf_pipeline = Pipeline([
    ("preprocess", build_preprocessor(numeric_features, categorical_features, scale_numeric=False)),
    ("clf", RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )),
])

OOF_CACHE = Path("data/processed/oof_predictions.npz")

if OOF_CACHE.exists():
    print(f"\nLoading cached OOF predictions from {OOF_CACHE}...")
    cached = np.load(OOF_CACHE)
    lr_oof, rf_oof = cached["lr_oof"], cached["rf_oof"]
else:
    print("\nComputing OOF predictions (Logistic Regression, 3-fold)...")
    t0 = time.time()
    lr_oof = cross_val_predict(lr_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    print(f"  done in {time.time() - t0:.1f}s")

    print("Computing OOF predictions (Random Forest, 3-fold)...")
    t0 = time.time()
    rf_oof = cross_val_predict(rf_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    print(f"  done in {time.time() - t0:.1f}s")

    OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OOF_CACHE, lr_oof=lr_oof, rf_oof=rf_oof)
    print(f"Cached OOF predictions to {OOF_CACHE}")

# --------------------------------------------------
# Step 18: meta-model on OOF predictions
# --------------------------------------------------
# class_weight="balanced" matters here just like for the base
# models - without it, the meta-model's decision boundary skews
# toward the majority class and recall (hence F1) collapses even
# though ranking metrics (ROC-AUC/PR-AUC) look fine.

meta_X_train = np.column_stack([lr_oof, rf_oof])
meta_model = LogisticRegression(class_weight="balanced")
meta_model.fit(meta_X_train, y_train)

print("\nMeta-model coefficients (LR weight, RF weight):", meta_model.coef_[0])
print("Meta-model intercept:", meta_model.intercept_[0])

# --------------------------------------------------
# Step 19: full base models (already fit on ALL of train,
# from 10_logistic_regression.py / 11_random_forest.py) generate
# val predictions, which the meta-model combines.
# --------------------------------------------------

lr_full = joblib.load("models/logistic_regression.joblib")
rf_full = joblib.load("models/random_forest.joblib")

lr_val_proba = lr_full.predict_proba(X_val)[:, 1]
rf_val_proba = rf_full.predict_proba(X_val)[:, 1]

meta_X_val = np.column_stack([lr_val_proba, rf_val_proba])
stacked_val_proba = meta_model.predict_proba(meta_X_val)[:, 1]
stacked_val_pred = (stacked_val_proba >= 0.5).astype(int)

print("\nValidation metrics (Stacked ensemble):")
print(f"  ROC-AUC:  {roc_auc_score(y_val, stacked_val_proba):.4f}")
print(f"  PR-AUC:   {average_precision_score(y_val, stacked_val_proba):.4f}")
print(f"  F1:       {f1_score(y_val, stacked_val_pred):.4f}")

Path("models").mkdir(exist_ok=True)
joblib.dump(meta_model, "models/meta_model.joblib")
print("\nSaved: models/meta_model.joblib")
