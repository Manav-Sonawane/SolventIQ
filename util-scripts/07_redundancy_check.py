import duckdb
import pandas as pd

DB_FILE = "data/processed/loan_data.duckdb"

con = duckdb.connect(DB_FILE)

numeric_columns = con.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'loans_clean'
      AND column_name != 'default_flag'
      AND data_type IN (
          'DOUBLE',
          'FLOAT',
          'INTEGER',
          'BIGINT',
          'DECIMAL'
      )
""").fetchdf()["column_name"].tolist()

# --------------------------------------------------
# A representative sample is fine here: this is an
# EDA-style redundancy check between features, not the
# target correlation itself (which was already computed
# on the full dataset in 06_correlation_audit.py).
# --------------------------------------------------

sample = con.execute(f"""
    SELECT {", ".join(f'"{c}"' for c in numeric_columns)}
    FROM loans_clean
    USING SAMPLE 300000
""").df()

corr_matrix = sample.corr()

pairs = []
for i, col_a in enumerate(numeric_columns):
    for col_b in numeric_columns[i + 1:]:
        value = corr_matrix.loc[col_a, col_b]
        if pd.notna(value) and abs(value) >= 0.85:
            pairs.append({
                "feature_a": col_a,
                "feature_b": col_b,
                "correlation": round(value, 4),
            })

pairs_df = pd.DataFrame(pairs).sort_values(
    "correlation", key=lambda s: s.abs(), ascending=False
).reset_index(drop=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)

print(f"Highly correlated feature pairs (|corr| >= 0.85), sample n={len(sample):,}:\n")
print(pairs_df)

con.close()
