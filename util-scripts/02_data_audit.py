import duckdb

DB_FILE = "data/processed/loan_data.duckdb"

con = duckdb.connect(DB_FILE)

# Number of rows
result = con.execute("""
    SELECT COUNT(*)
    FROM loans_clean
""").fetchone()

print("Rows:", result[0])

# Number of columns
result = con.execute("""
    SELECT COUNT(*)
    FROM information_schema.columns
    WHERE table_name = 'loans_clean'
""").fetchone()

print("Columns:", result[0])

# Target distribution
print("\nTarget distribution:")

print(
    con.execute("""
        SELECT
            default_flag,
            COUNT(*) AS count,
            ROUND(
                100.0 * COUNT(*) /
                SUM(COUNT(*)) OVER (),
                2
            ) AS percentage
        FROM loans_clean
        GROUP BY default_flag
        ORDER BY default_flag
    """).df()
)

con.close()
