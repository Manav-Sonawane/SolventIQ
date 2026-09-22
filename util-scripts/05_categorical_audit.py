import duckdb
import pandas as pd

DB_FILE = "data/processed/loan_data.duckdb"

con = duckdb.connect(DB_FILE)

categorical_columns = con.execute("""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'loans_clean'
      AND column_name NOT IN ('loan_status', 'default_flag')
      AND data_type IN ('VARCHAR', 'BOOLEAN')
    ORDER BY ordinal_position
""").fetchdf()["column_name"].tolist()

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)

for column in categorical_columns:

    result = con.execute(f"""
        SELECT
            "{column}" AS value,
            COUNT(*) AS loans,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total,
            ROUND(AVG(default_flag), 4) AS default_rate
        FROM loans_clean
        GROUP BY "{column}"
        ORDER BY loans DESC
    """).df()

    n_distinct = len(result)

    print(f"\n=== {column} ({n_distinct} distinct values) ===")

    if n_distinct > 30:
        print(result.head(15))
        print(f"... ({n_distinct - 15} more values not shown)")
    else:
        print(result)

con.close()
