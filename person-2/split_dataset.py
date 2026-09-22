import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "working_dataset.csv"

print("Loading working dataset...")

df = pd.read_csv(INPUT)

# Convert target: No -> 0, Yes -> 1
df["Is Fraud?"] = df["Is Fraud?"].map({
    "No": 0,
    "Yes": 1
})

print("\nOriginal distribution:")
print(df["Is Fraud?"].value_counts())

# 70% training, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["Is Fraud?"],
    random_state=42
)

# Split remaining 30% into:
# 15% validation + 15% test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["Is Fraud?"],
    random_state=42
)

# Save each dataset
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\n========== SPLIT COMPLETE ==========")

print(f"Training   : {len(train_df):,}")
print(f"Validation : {len(val_df):,}")
print(f"Test       : {len(test_df):,}")

print("\n========== FRAUD COUNTS ==========")

print("\nTraining:")
print(train_df["Is Fraud?"].value_counts())

print("\nValidation:")
print(val_df["Is Fraud?"].value_counts())

print("\nTest:")
print(test_df["Is Fraud?"].value_counts())