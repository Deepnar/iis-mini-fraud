import pandas as pd

FILE = r"C:\Users\iamya\.cache\kagglehub\datasets\ealtman2019\credit-card-transactions\versions\8\credit_card_transactions-ibm_v2.csv"

fraud_count = 0
legit_count = 0
total = 0

print("Scanning dataset...")

for chunk in pd.read_csv(
    FILE,
    usecols=["Is Fraud?"],
    chunksize=500_000
):
    counts = chunk["Is Fraud?"].value_counts()

    fraud_count += counts.get("Yes", 0)
    legit_count += counts.get("No", 0)
    total += len(chunk)

print("\n========== DATASET SUMMARY ==========")
print(f"Total transactions : {total:,}")
print(f"Legitimate         : {legit_count:,}")
print(f"Fraud              : {fraud_count:,}")
print(f"Fraud percentage   : {(fraud_count / total) * 100:.4f}%")
print("====================================")