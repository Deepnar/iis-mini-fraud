import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# =========================
# Load training + test data
# =========================

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()


# =========================
# Final XGBoost model
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
# Train on training data
# =========================

print("Training final XGBoost model...")

model.fit(X_train, y_train)


# =========================
# Predict on TEST data
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# Calculate metrics
# =========================

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
pr_auc = average_precision_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)


# =========================
# Display results
# =========================

print("\n==============================")
print("FINAL XGBOOST TEST RESULTS")
print("==============================")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))