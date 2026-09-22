import duckdb
import pandas as pd

DB_FILE = "data/processed/loan_data.duckdb"

con = duckdb.connect(DB_FILE)

chunk_size = 100_000

chunk_number = 0
total_rows = 0

while True:

    chunk = con.execute(f"""
        SELECT *
        FROM loans_clean
        LIMIT {chunk_size}
        OFFSET {total_rows}
    """).df()

    if chunk.empty:
        break

    chunk_number += 1
    total_rows += len(chunk)

    print(
        f"Chunk {chunk_number}: "
        f"{len(chunk):,} rows"
    )

    # Example Pandas operation
    print(
        "Default rate:",
        chunk["default_flag"].mean()
    )

print("\nTotal processed:", total_rows)

con.close()
