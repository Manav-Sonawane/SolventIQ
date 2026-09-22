import duckdb
import pandas as pd

DB_FILE = "data/processed/loan_data.duckdb"

con = duckdb.connect(DB_FILE)

total_rows = con.execute("SELECT COUNT(*) FROM loans_clean").fetchone()[0]

columns = con.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'loans_clean'
    ORDER BY ordinal_position
""").fetchdf()["column_name"].tolist()

# --------------------------------------------------
# Exact null counts for every column, in one query
# --------------------------------------------------

select_sql = ",\n    ".join(
    f'COUNT(*) - COUNT("{col}") AS "{col}"' for col in columns
)

query = f"""
SELECT
    {select_sql}
FROM loans_clean
"""

row = con.execute(query).fetchone()

missing_df = pd.DataFrame({
    "feature": columns,
    "nulls": row,
})

missing_df["total_rows"] = total_rows
missing_df["missing_pct"] = (
    100.0 * missing_df["nulls"] / total_rows
).round(2)

missing_df = missing_df.sort_values(
    "missing_pct", ascending=False
).reset_index(drop=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)

print(f"Total rows: {total_rows}\n")
print(missing_df[["feature", "nulls", "missing_pct"]])

con.close()
