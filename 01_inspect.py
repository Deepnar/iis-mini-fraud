"""Person 1 - Step 1: chunked inspection of IBM v2 CSV (2.3GB / 24M rows).
Run: uv run python 01_inspect.py
Does NOT load full file into RAM.
"""
import os
import pandas as pd

import kagglehub

DATA_DIR = kagglehub.dataset_download("ealtman2019/credit-card-transactions")
MAIN = os.path.join(DATA_DIR, "credit_card_transactions-ibm_v2.csv")

print("DATA_DIR:", DATA_DIR)

# 1. schema from small sample
sample = pd.read_csv(MAIN, nrows=5)
print("\n--- COLUMNS ---")
print(list(sample.columns))
print(sample.head(2).to_string())
print("\n--- DTYPES (sample) ---")
print(sample.dtypes)

# 2. full scan in chunks: counts, fraud ratio, missing, cardinalities
usecols = ["User", "Card", "Amount", "Use Chip", "Merchant Name",
           "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?"]
chunks = pd.read_csv(MAIN, usecols=usecols, chunksize=500_000)

total = 0
fraud = 0
missing = None
states = set()
mcc_n = set()
use_chip_vals = set()
errors_vals = set()

for i, c in enumerate(chunks):
    total += len(c)
    fraud += (c["Is Fraud?"] == "Yes").sum()
    m = c.isna().sum()
    missing = m if missing is None else missing + m
    states.update(c["Merchant State"].dropna().unique().tolist()[:1000])
    mcc_n.update(c["MCC"].dropna().unique().tolist())
    use_chip_vals.update(c["Use Chip"].dropna().unique().tolist())
    errors_vals.update(c["Errors?"].dropna().unique().tolist())
    print(f"chunk {i+1}: rows={total:,} fraud_so_far={fraud:,}")

print("\n=== SUMMARY ===")
print(f"total transactions : {total:,}")
print(f"fraud              : {fraud:,} ({fraud/total*100:.4f}%)")
print(f"legit              : {total-fraud:,}")
print("\nmissing values:\n", missing)
print("\nMerchant State cardinality:", len(states))
print("MCC cardinality:", len(mcc_n))
print("Use Chip values:", sorted(map(str, use_chip_vals)))
print("Errors? values:", sorted(map(str, errors_vals)))
print("\nTarget column: 'Is Fraud?' (Yes/No) -> map to 1/0 in modeling.")
print("Leakage rule: split first (stratified), fit scalers/encoders on TRAIN only.")
