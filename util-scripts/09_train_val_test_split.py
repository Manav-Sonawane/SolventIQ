import duckdb

DB_FILE = "data/processed/loan_data.duckdb"

con = duckdb.connect(DB_FILE)

print("Creating time-based train/val/test split...")

# --------------------------------------------------
# Time-based split (not random): earlier-issued loans
# train the model, later-issued loans validate/test it.
# This matches how the model will actually be used -
# train on historical borrowers, predict future ones -
# and avoids leaking future credit-market conditions
# (e.g. 2016+ underwriting shifts) into training.
#
# Split points: 70% train / 15% val / 15% test, by
# cumulative loan-count over issue_d (not by year,
# since loan volume grew ~1000x from 2007 to 2015).
# --------------------------------------------------

con.execute("""
    CREATE OR REPLACE TABLE loans_split AS
    SELECT
        *,
        STRPTIME(issue_d, '%b-%Y') AS issue_date,
        PERCENT_RANK() OVER (ORDER BY STRPTIME(issue_d, '%b-%Y')) AS pct_rank
    FROM loans_model
""")

con.execute("""
    ALTER TABLE loans_split ADD COLUMN split VARCHAR
""")

con.execute("""
    UPDATE loans_split
    SET split = CASE
        WHEN pct_rank < 0.70 THEN 'train'
        WHEN pct_rank < 0.85 THEN 'val'
        ELSE 'test'
    END
""")

summary = con.execute("""
    SELECT
        split,
        COUNT(*) AS rows,
        MIN(issue_date) AS min_issue,
        MAX(issue_date) AS max_issue,
        ROUND(AVG(default_flag), 4) AS default_rate
    FROM loans_split
    GROUP BY split
    ORDER BY min_issue
""").df()

print("\nSplit summary:\n")
print(summary)

con.close()
