import pandas as pd

FILE = r"C:\Users\iamya\.cache\kagglehub\datasets\ealtman2019\credit-card-transactions\versions\8\credit_card_transactions-ibm_v2.csv"

print("Reading first 500,000 rows...")

df = pd.read_csv(FILE, nrows=500_000)

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== UNIQUE VALUES ==========")
for col in df.columns:
    print(f"{col:20} : {df[col].nunique():,}")

print("\n========== SAMPLE ==========")
print(df.head())

print("\n========== FRAUD DISTRIBUTION IN SAMPLE ==========")
print(df["Is Fraud?"].value_counts())

print("\n========== FRAUD PERCENTAGE IN SAMPLE ==========")
fraud_count = (df["Is Fraud?"] == "Yes").sum()
total = len(df)

print(f"Fraud: {fraud_count:,}")
print(f"Legitimate: {total - fraud_count:,}")
print(f"Fraud percentage: {(fraud_count / total) * 100:.4f}%")