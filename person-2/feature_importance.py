import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt


# =========================
# Load training data
# =========================

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()


# =========================
# Train our best XGBoost model
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

model.fit(X_train, y_train)


# =========================
# Get feature importance
# =========================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# =========================
# Display
# =========================

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")

print(feature_importance)


# =========================
# Plot top features
# =========================

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")

plt.tight_layout()
plt.savefig("xgboost_feature_importance.png", dpi=300)

plt.show()


# =========================
# Save results
# =========================

feature_importance.to_csv(
    "xgboost_feature_importance.csv",
    index=False
)

print("\nFeature importance saved.")