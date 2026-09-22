import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

print("Loading datasets...")

train = pd.read_csv("train.csv")
validation = pd.read_csv("validation.csv")
test = pd.read_csv("test.csv")


# ==========================================
# 1. Convert Amount to numeric
# ==========================================

for df in [train, validation, test]:
    df["Amount"] = (
        df["Amount"]
        .str.replace("$", "", regex=False)
        .astype(float)
    )


# ==========================================
# 2. Convert Time into useful numerical values
# ==========================================

for df in [train, validation, test]:

    time_parts = df["Time"].str.split(":", expand=True)

    df["Hour"] = time_parts[0].astype(int)
    df["Minute"] = time_parts[1].astype(int)

    df["TimeMinutes"] = (
        df["Hour"] * 60 + df["Minute"]
    )


# ==========================================
# 3. Handle missing categorical values
# ==========================================

categorical_columns = [
    "Use Chip",
    "Merchant City",
    "Merchant State",
    "Errors?"
]

for df in [train, validation, test]:
    for col in categorical_columns:
        df[col] = df[col].fillna("Unknown")


# ==========================================
# 4. Encode categorical columns
# ==========================================
# IMPORTANT:
# Encoder learns mappings ONLY from training data.
# Validation/test use the same mappings.

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

train[categorical_columns] = encoder.fit_transform(
    train[categorical_columns]
)

validation[categorical_columns] = encoder.transform(
    validation[categorical_columns]
)

test[categorical_columns] = encoder.transform(
    test[categorical_columns]
)


# ==========================================
# 5. Handle missing Zip
# ==========================================

for df in [train, validation, test]:
    df["Zip"] = df["Zip"].fillna(-1)


# ==========================================
# 6. Convert target to 0 / 1
# ==========================================

for df in [train, validation, test]:
    df["Is Fraud?"] = df["Is Fraud?"].astype(int)


# ==========================================
# 7. Remove unnecessary columns
# ==========================================

drop_columns = [
    "Time",
    "Hour",
    "Minute",
    "Is Fraud?"
]


X_train = train.drop(columns=drop_columns)
y_train = train["Is Fraud?"]

X_validation = validation.drop(columns=drop_columns)
y_validation = validation["Is Fraud?"]

X_test = test.drop(columns=drop_columns)
y_test = test["Is Fraud?"]


# ==========================================
# 8. Save processed datasets
# ==========================================

X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)

X_validation.to_csv("X_validation.csv", index=False)
y_validation.to_csv("y_validation.csv", index=False)

X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)


# ==========================================
# 9. Display results
# ==========================================

print("\n========== PREPROCESSING COMPLETE ==========")

print("X_train      :", X_train.shape)
print("X_validation :", X_validation.shape)
print("X_test       :", X_test.shape)

print("\nFeatures:")
print(list(X_train.columns))

print("\nData types:")
print(X_train.dtypes)

print("\nTraining target:")
print(y_train.value_counts())

print("\nValidation target:")
print(y_validation.value_counts())

print("\nTest target:")
print(y_test.value_counts())