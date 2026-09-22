import duckdb

DB_FILE = "data/processed/loan_data.duckdb"

con = duckdb.connect(DB_FILE)

print("Building engineered feature table (loans_model)...")

# --------------------------------------------------
# Feature engineering + redundancy trim, all in DuckDB
# --------------------------------------------------
#
# Engineered:
#   fico_mid              = (fico_range_low + fico_range_high) / 2
#   term_months            = "36 months" -> 36 (int)
#   emp_length_years       = "< 1 year".."10+ years" -> 0..10 (int, NULL stays NULL)
#   credit_history_years   = years between earliest_cr_line and issue_d
#   never_delinquent       = 1 if mths_since_last_delinq is NULL, else 0
#                             (missing here means "no delinquency on record",
#                             not missing data - see IMPLEMENTATION.md step 15)
#
# Dropped as redundant (see 07_redundancy_check.py, |corr| >= 0.85 pairs):
#   funded_amnt, funded_amnt_inv   -> near-duplicate of loan_amnt
#   fico_range_low, fico_range_high -> replaced by fico_mid
#   num_sats                        -> near-duplicate of open_acc
#   num_rev_tl_bal_gt_0             -> near-duplicate of num_actv_rev_tl
#   total_il_high_credit_limit      -> near-duplicate of total_bal_il
#
# Dropped as high-cardinality / low marginal value:
#   zip_code       -> 944 distinct values, addr_state already captures geography
#   sub_grade      -> 35 distinct values, int_rate already prices risk finely
#
# Dropped as redundant with application_type (98% missing, joint-only fields):
#   annual_inc_joint, dti_joint, verification_status_joint
#
# Kept for splitting only, not as a model feature:
#   issue_d  -> used for the time-based train/val/test split (step 19),
#               excluded from loans_model's feature set below
#
# Replaced by engineered equivalents (raw form dropped):
#   fico_range_low, fico_range_high, term, emp_length,
#   earliest_cr_line

query = """
CREATE OR REPLACE TABLE loans_model AS

SELECT
    issue_d,

    -- engineered
    (fico_range_low + fico_range_high) / 2.0 AS fico_mid,

    CAST(SPLIT_PART(TRIM(term), ' ', 1) AS INTEGER) AS term_months,

    CASE
        WHEN emp_length IS NULL THEN NULL
        WHEN emp_length = '< 1 year' THEN 0
        WHEN emp_length = '10+ years' THEN 10
        ELSE CAST(REGEXP_EXTRACT(emp_length, '(\\d+)', 1) AS INTEGER)
    END AS emp_length_years,

    DATE_DIFF(
        'day',
        STRPTIME(earliest_cr_line, '%b-%Y'),
        STRPTIME(issue_d, '%b-%Y')
    ) / 365.25 AS credit_history_years,

    CASE WHEN mths_since_last_delinq IS NULL THEN 1 ELSE 0 END AS never_delinquent,
    mths_since_last_delinq,

    -- kept as-is
    loan_amnt,
    int_rate,
    installment,
    grade,
    home_ownership,
    annual_inc,
    verification_status,
    purpose,
    addr_state,
    dti,
    delinq_2yrs,
    inq_last_6mths,
    open_acc,
    pub_rec,
    revol_bal,
    revol_util,
    total_acc,
    application_type,
    acc_now_delinq,
    tot_coll_amt,
    tot_cur_bal,
    open_acc_6m,
    open_act_il,
    open_il_12m,
    open_il_24m,
    total_bal_il,
    il_util,
    open_rv_12m,
    open_rv_24m,
    max_bal_bc,
    all_util,
    total_rev_hi_lim,
    inq_fi,
    total_cu_tl,
    inq_last_12m,
    acc_open_past_24mths,
    avg_cur_bal,
    bc_open_to_buy,
    bc_util,
    delinq_amnt,
    mort_acc,
    mths_since_recent_bc,
    mths_since_recent_inq,
    num_accts_ever_120_pd,
    num_actv_bc_tl,
    num_actv_rev_tl,
    num_bc_sats,
    num_bc_tl,
    num_il_tl,
    num_op_rev_tl,
    num_rev_accts,
    num_tl_30dpd,
    num_tl_90g_dpd_24m,
    num_tl_op_past_12m,
    pct_tl_nvr_dlq,
    percent_bc_gt_75,
    pub_rec_bankruptcies,
    tax_liens,
    tot_hi_cred_lim,
    total_bal_ex_mort,
    total_bc_limit,

    default_flag

FROM loans_clean
"""

con.execute(query)

result = con.execute("""
    SELECT COUNT(*), COUNT(*) - COUNT(term_months), COUNT(*) - COUNT(fico_mid),
           COUNT(*) - COUNT(credit_history_years)
    FROM loans_model
""").fetchone()

n_cols = con.execute("""
    SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'loans_model'
""").fetchone()[0]

print("\nloans_model created.")
print("Rows:", result[0])
print("Columns:", n_cols)
print("Null term_months:", result[1])
print("Null fico_mid:", result[2])
print("Null credit_history_years:", result[3])

con.close()
