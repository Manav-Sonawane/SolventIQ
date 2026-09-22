import duckdb
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_FILE = Path("data/raw/accepted_loans.csv")
DB_FILE = Path("data/processed/loan_data.duckdb")

DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Connect
# --------------------------------------------------

con = duckdb.connect(str(DB_FILE))

print("Creating cleaned loan database...")

# --------------------------------------------------
# Columns we want
# --------------------------------------------------

features = [
    "issue_d",
    "loan_amnt",
    "funded_amnt",
    "funded_amnt_inv",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "zip_code",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "total_rev_hi_lim",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "delinq_amnt",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit"
]

# --------------------------------------------------
# Convert list into SQL
# --------------------------------------------------

feature_sql = ",\n        ".join(features)

# --------------------------------------------------
# Create cleaned table
# --------------------------------------------------

query = f"""
CREATE OR REPLACE TABLE loans_clean AS

SELECT
        {feature_sql},

        loan_status,

        CASE
            WHEN loan_status = 'Charged Off' THEN 1
            WHEN loan_status = 'Fully Paid' THEN 0
        END AS default_flag

FROM read_csv_auto(
    '{RAW_FILE}',
    header = true,
    ignore_errors = true
)

WHERE loan_status IN (
    'Fully Paid',
    'Charged Off'
);
"""

con.execute(query)

# --------------------------------------------------
# Check result
# --------------------------------------------------

result = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        SUM(default_flag) AS defaults,
        COUNT(*) - SUM(default_flag) AS non_defaults
    FROM loans_clean
""").fetchone()

print("\nDatabase created successfully.")

print("Total rows:", result[0])
print("Defaults:", result[1])
print("Non-defaults:", result[2])

print("\nDatabase:", DB_FILE)

con.close()
