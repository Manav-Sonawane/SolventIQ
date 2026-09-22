import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_FILE = "data/processed/loan_data.duckdb"

TARGET = "default_flag"

CATEGORICAL_FEATURES = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

# "months since X" fields where NULL means "X never happened",
# not missing data - see IMPLEMENTATION.md step 15 and the
# missingness audit (mths_since_last_delinq is 50% null).
# Filled with a large sentinel (never happened == a very long time ago)
# instead of a median, which would wrongly imply recent activity.
MTHS_SINCE_SENTINEL_FEATURES = [
    "mths_since_last_delinq",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
]
MTHS_SINCE_SENTINEL_VALUE = 999

_EXCLUDED = {TARGET, "issue_d", "issue_date", "pct_rank", "split"}


def load_split(con=None):
    """Load train/val/test as pandas DataFrames from loans_split."""
    own_conn = con is None
    if own_conn:
        con = duckdb.connect(DB_FILE, read_only=True)

    df = con.execute("SELECT * FROM loans_split").df()

    if own_conn:
        con.close()

    for col in MTHS_SINCE_SENTINEL_FEATURES:
        df[col] = df[col].fillna(MTHS_SINCE_SENTINEL_VALUE)

    feature_cols = [c for c in df.columns if c not in _EXCLUDED]
    numeric_features = [
        c for c in feature_cols
        if c not in CATEGORICAL_FEATURES and c != TARGET
    ]

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    return train, val, test, numeric_features, CATEGORICAL_FEATURES


def build_preprocessor(numeric_features, categorical_features, scale_numeric):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def Xy(df, numeric_features, categorical_features):
    X = df[numeric_features + categorical_features]
    y = df[TARGET]
    return X, y
