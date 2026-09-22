import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


# =========================
# Load data
# =========================

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

X_validation = pd.read_csv("X_validation.csv")
y_validation = pd.read_csv("y_validation.csv").squeeze()


# =========================
# Different XGBoost settings
# =========================

experiments = [
    {
        "name": "Tuning 1",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    {
        "name": "Tuning 2",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    {
        "name": "Tuning 3",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    },
    {
        "name": "Tuning 4",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9
    }
]


results = []


# =========================
# Run experiments
# =========================

for exp in experiments:

    print("\n==============================")
    print(exp["name"])
    print("==============================")

    model = xgb.XGBClassifier(
        n_estimators=exp["n_estimators"],
        max_depth=exp["max_depth"],
        learning_rate=exp["learning_rate"],
        subsample=exp["subsample"],
        colsample_bytree=exp["colsample_bytree"],
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_validation)
    y_prob = model.predict_proba(X_validation)[:, 1]

    precision = precision_score(y_validation, y_pred)
    recall = recall_score(y_validation, y_pred)
    f1 = f1_score(y_validation, y_pred)
    pr_auc = average_precision_score(y_validation, y_prob)

    cm = confusion_matrix(y_validation, y_pred)

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("PR-AUC:", round(pr_auc, 4))

    print("Confusion Matrix:")
    print(cm)

    results.append({
        "Experiment": exp["name"],
        "n_estimators": exp["n_estimators"],
        "max_depth": exp["max_depth"],
        "learning_rate": exp["learning_rate"],
        "subsample": exp["subsample"],
        "colsample_bytree": exp["colsample_bytree"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "PR-AUC": pr_auc
    })


# =========================
# Save results
# =========================

tuning_results = pd.DataFrame(results)

tuning_results.to_csv(
    "xgboost_tuning_results.csv",
    index=False
)

print("\n==============================")
print("TUNING RESULTS")
print("==============================")

print(tuning_results)