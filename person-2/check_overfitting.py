import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)


# =========================
# Load data
# =========================

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

X_validation = pd.read_csv("X_validation.csv")
y_validation = pd.read_csv("y_validation.csv").squeeze()


# =========================
# Tuning 4 model
# =========================

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)


# =========================
# Train
# =========================

model.fit(X_train, y_train)


# =========================
# TRAINING PERFORMANCE
# =========================

train_pred = model.predict(X_train)
train_prob = model.predict_proba(X_train)[:, 1]

train_precision = precision_score(y_train, train_pred)
train_recall = recall_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)
train_pr_auc = average_precision_score(y_train, train_prob)


# =========================
# VALIDATION PERFORMANCE
# =========================

val_pred = model.predict(X_validation)
val_prob = model.predict_proba(X_validation)[:, 1]

val_precision = precision_score(y_validation, val_pred)
val_recall = recall_score(y_validation, val_pred)
val_f1 = f1_score(y_validation, val_pred)
val_pr_auc = average_precision_score(y_validation, val_prob)


# =========================
# Display results
# =========================

print("\n==============================")
print("TRAINING PERFORMANCE")
print("==============================")

print("Precision:", round(train_precision, 4))
print("Recall:", round(train_recall, 4))
print("F1:", round(train_f1, 4))
print("PR-AUC:", round(train_pr_auc, 4))


print("\n==============================")
print("VALIDATION PERFORMANCE")
print("==============================")

print("Precision:", round(val_precision, 4))
print("Recall:", round(val_recall, 4))
print("F1:", round(val_f1, 4))
print("PR-AUC:", round(val_pr_auc, 4))


# =========================
# Compare
# =========================

print("\n==============================")
print("GENERALIZATION GAP")
print("==============================")

print(
    "F1 Gap:",
    round(train_f1 - val_f1, 4)
)

print(
    "PR-AUC Gap:",
    round(train_pr_auc - val_pr_auc, 4)
)