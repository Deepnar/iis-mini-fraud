import pandas as pd
import numpy as np

FILE = r"C:\Users\iamya\.cache\kagglehub\datasets\ealtman2019\credit-card-transactions\versions\8\credit_card_transactions-ibm_v2.csv"

OUTPUT = "working_dataset.csv"

TARGET_ROWS = 1_000_000
CHUNK_SIZE = 500_000

print("Starting dataset preparation...")

rng = np.random.default_rng(42)

samples = []
total_rows = 0

for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE):

    total_rows += len(chunk)

    # Randomly sample approximately 4% from each chunk
    sample_size = int(len(chunk) * TARGET_ROWS / 24_386_900)

    sampled = chunk.sample(
        n=sample_size,
        random_state=42
    )

    samples.append(sampled)

    print(
        f"Processed {total_rows:,} rows | "
        f"Collected {sum(len(x) for x in samples):,}"
    )

df = pd.concat(samples, ignore_index=True)

# Shuffle the final dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv(OUTPUT, index=False)

print("\n========== DATASET CREATED ==========")
print(f"Rows       : {len(df):,}")
print(f"Columns    : {len(df.columns)}")

print("\n========== TARGET DISTRIBUTION ==========")
print(df["Is Fraud?"].value_counts())

fraud = (df["Is Fraud?"] == "Yes").sum()

print(f"\nFraud percentage: {(fraud / len(df)) * 100:.4f}%")
print(f"Saved to: {OUTPUT}")