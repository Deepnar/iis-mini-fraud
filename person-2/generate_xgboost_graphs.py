import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# XGBoost PROJECT — PRESENTATION GRAPHS
# Generates 4 graphs from the experiments already completed.
# ============================================================

# ------------------------------------------------------------
# 1. Experiment comparison
# ------------------------------------------------------------

results = pd.DataFrame({
    "Experiment": [
        "Baseline",
        "Class Weighting",
        "Tuning 1",
        "Tuning 2",
        "Tuning 3",
        "Tuning 4"
    ],
    "Precision": [0.9570, 0.0668, 0.9706, 0.9545, 0.9468, 0.9798],
    "Recall":    [0.4890, 0.8077, 0.3626, 0.4615, 0.4890, 0.5330],
    "F1":        [0.6473, 0.1235, 0.5280, 0.6222, 0.6449, 0.6904],
    "PR-AUC":    [0.6857, 0.6077, 0.6263, 0.6826, 0.6910, 0.7020]
})

plt.figure(figsize=(11, 6))
x = np.arange(len(results))
width = 0.20

plt.bar(x - 1.5 * width, results["Precision"], width, label="Precision")
plt.bar(x - 0.5 * width, results["Recall"], width, label="Recall")
plt.bar(x + 0.5 * width, results["F1"], width, label="F1")
plt.bar(x + 1.5 * width, results["PR-AUC"], width, label="PR-AUC")

plt.xticks(x, results["Experiment"], rotation=20)
plt.ylabel("Score")
plt.xlabel("Experiment")
plt.title("XGBoost Experiment Comparison")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("01_xgboost_experiment_comparison.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 2. Final test-set confusion matrix
# ------------------------------------------------------------

cm = np.array([
    [149805, 8],
    [70, 112]
])

plt.figure(figsize=(7, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Final XGBoost — Test Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.yticks([0, 1], ["Legitimate", "Fraud"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")

plt.colorbar(label="Number of Transactions")
plt.tight_layout()
plt.savefig("02_final_xgboost_confusion_matrix.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 3. Precision vs Recall
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    results["Recall"],
    results["Precision"],
    marker="o"
)

for _, row in results.iterrows():
    plt.annotate(
        row["Experiment"],
        (row["Recall"], row["Precision"]),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("XGBoost Precision vs Recall")
plt.xlim(0, 0.9)
plt.ylim(0, 1.05)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("03_xgboost_precision_vs_recall.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 4. Training vs Validation — generalization gap
# ------------------------------------------------------------

metrics = ["F1", "PR-AUC"]
training = [0.8587, 0.9523]
validation = [0.6904, 0.7020]

x = np.arange(len(metrics))
width = 0.32

plt.figure(figsize=(8, 6))

plt.bar(x - width / 2, training, width, label="Training")
plt.bar(x + width / 2, validation, width, label="Validation")

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("XGBoost Generalization: Training vs Validation")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("04_xgboost_generalization_gap.png", dpi=300)
plt.show()


print("\n========================================")
print("ALL 4 GRAPHS GENERATED SUCCESSFULLY")
print("========================================")
print("1. 01_xgboost_experiment_comparison.png")
print("2. 02_final_xgboost_confusion_matrix.png")
print("3. 03_xgboost_precision_vs_recall.png")
print("4. 04_xgboost_generalization_gap.png")
print("\nThese files are ready to use in your project presentation/report.")
