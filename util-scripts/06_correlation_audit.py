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
# Compute all correlations against default_flag
# in a single DuckDB query (full dataset, no sampling)
# --------------------------------------------------

select_sql = ",\n    ".join(
    f'corr("{col}", default_flag) AS "{col}"' for col in numeric_columns
)

query = f"""
SELECT
    {select_sql}
FROM loans_clean
"""

row = con.execute(query).fetchone()

correlation_df = pd.DataFrame({
    "feature": numeric_columns,
    "correlation": row,
})

correlation_df["abs_correlation"] = correlation_df["correlation"].abs()

correlation_df = correlation_df.sort_values(
    "abs_correlation", ascending=False
).reset_index(drop=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)

print(correlation_df[["feature", "correlation"]])

con.close()
