import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

print("Loading processed datasets...")

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

X_validation = pd.read_csv("X_validation.csv")
y_validation = pd.read_csv("y_validation.csv").squeeze()


# ==========================================
# EXPERIMENT 2 - CLASS WEIGHTING
# ==========================================

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,

    # Handle class imbalance
    scale_pos_weight=822.5,

    random_state=42,
    eval_metric="logloss"
)


# ==========================================
# TRAIN
# ==========================================

print("\nTraining XGBoost with class weighting...")

model.fit(
    X_train,
    y_train
)


# ==========================================
# VALIDATION PREDICTIONS
# ==========================================

print("\nMaking validation predictions...")

y_pred = model.predict(X_validation)
y_prob = model.predict_proba(X_validation)[:, 1]


# ==========================================
# METRICS
# ==========================================

precision = precision_score(
    y_validation,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_validation,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_validation,
    y_pred,
    zero_division=0
)

pr_auc = average_precision_score(
    y_validation,
    y_prob
)

cm = confusion_matrix(
    y_validation,
    y_pred
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n========== EXPERIMENT 2 RESULTS ==========")

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\n===========================================")


# ==========================================
# SAVE RESULT WITHOUT DELETING EXPERIMENT 1
# ==========================================

new_result = pd.DataFrame([
    {
        "Experiment": "XGBoost + Class Weighting",
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc
    }
])

try:
    existing_results = pd.read_csv("experiment_results.csv")

    existing_results = existing_results[
        existing_results["Experiment"] != "XGBoost + Class Weighting"
    ]

    all_results = pd.concat(
        [existing_results, new_result],
        ignore_index=True
    )

except FileNotFoundError:
    all_results = new_result

all_results.to_csv(
    "experiment_results.csv",
    index=False
)

print("\nExperiment 2 saved.")
print("\nCurrent experiment results:")
print(all_results)