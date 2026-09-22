import pandas as pd


# ============================================================
# PERSON 2 — XGBOOST EXPERIMENT OUTPUT
# ============================================================

results = [

    {
        "experiment": "A_baseline",
        "precision_fraud": 0.9570,
        "recall_fraud": 0.4890,
        "f1_fraud": 0.6473,
        "pr_auc": 0.6857,
        "TN": 149808,
        "FP": 4,
        "FN": 93,
        "TP": 89
    },

    {
        "experiment": "B_class_weighting",
        "precision_fraud": 0.0668,
        "recall_fraud": 0.8077,
        "f1_fraud": 0.1235,
        "pr_auc": 0.6077,
        "TN": 147760,
        "FP": 2052,
        "FN": 35,
        "TP": 147
    },

    {
        "experiment": "C_tuning_1",
        "precision_fraud": 0.9706,
        "recall_fraud": 0.3626,
        "f1_fraud": 0.5280,
        "pr_auc": 0.6263,
        "TN": 149810,
        "FP": 2,
        "FN": 116,
        "TP": 66
    },

    {
        "experiment": "D_tuning_2",
        "precision_fraud": 0.9545,
        "recall_fraud": 0.4615,
        "f1_fraud": 0.6222,
        "pr_auc": 0.6826,
        "TN": 149808,
        "FP": 4,
        "FN": 98,
        "TP": 84
    },

    {
        "experiment": "E_tuning_3",
        "precision_fraud": 0.9468,
        "recall_fraud": 0.4890,
        "f1_fraud": 0.6449,
        "pr_auc": 0.6910,
        "TN": 149807,
        "FP": 5,
        "FN": 93,
        "TP": 89
    },

    {
        "experiment": "F_tuning_4",
        "precision_fraud": 0.9798,
        "recall_fraud": 0.5330,
        "f1_fraud": 0.6904,
        "pr_auc": 0.7020,
        "TN": 149810,
        "FP": 2,
        "FN": 85,
        "TP": 97
    },

    {
        "experiment": "G_final_test",
        "precision_fraud": 0.9333,
        "recall_fraud": 0.6154,
        "f1_fraud": 0.7417,
        "pr_auc": 0.7701,
        "TN": 149805,
        "FP": 8,
        "FN": 70,
        "TP": 112
    }
]


# ============================================================
# CREATE TABLE
# ============================================================

df = pd.DataFrame(results)


# Round metric columns

metric_columns = [
    "precision_fraud",
    "recall_fraud",
    "f1_fraud",
    "pr_auc"
]

df[metric_columns] = df[metric_columns].round(4)


# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("PERSON 2 — XGBOOST RESULTS")
print("==============================================\n")

print(df.to_string(index=False))


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    "output_person2_xgboost.csv",
    index=False
)


print("\n==============================================")
print("Output saved as:")
print("output_person2_xgboost.csv")
print("==============================================")