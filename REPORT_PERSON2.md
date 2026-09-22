# Credit Card Fraud Detection — XGBoost

## 1. Objective

The objective is to develop an XGBoost model for credit card fraud detection and evaluate its performance under severe class imbalance. The experiments cover baseline training, class weighting, hyperparameter tuning, feature importance, and final test evaluation.

---

## 2. Dataset

The IBM Synthetic Credit Card Transactions dataset contains **24,386,900 transactions**, including **29,757 fraud transactions (0.1220%)**.

For experimentation, approximately **1 million transactions** were sampled.

| Class | Rows |
|---|---:|
| Legitimate | 998,747 |
| Fraud | 1,214 |
| Total | 999,961 |

The data was split using a stratified **70/15/15** split.

| Split | Rows | Fraud |
|---|---:|---:|
| Train | 699,972 | 850 |
| Validation | 149,994 | 182 |
| Test | 149,995 | 182 |

The test set was kept separate until final evaluation.

---

## 3. Method

Preprocessing included:

- Converting `Amount` to numeric values.
- Extracting `TimeMinutes` from `Time`.
- Handling missing values.
- Encoding categorical features using `OrdinalEncoder`.

The model used 14 transaction features including merchant, transaction, geographic and time-related fields.

XGBoost was evaluated through:
1. Baseline model.
2. Class-weighting experiment.
3. Four hyperparameter configurations.
4. Feature-importance analysis.
5. Final test evaluation.

---

## 4. Experiments and Results

### 4.1 Baseline XGBoost

Configuration:

```text
n_estimators = 100
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
```

Validation results:

| Precision | Recall | F1 | PR-AUC |
|---:|---:|---:|---:|
| 0.9570 | 0.4890 | 0.6473 | 0.6857 |

The model detected 89 of 182 fraud cases with only 4 false positives.

---

### 4.2 Class Weighting

`scale_pos_weight = 822.5` was tested to give greater importance to the fraud class.

| Precision | Recall | F1 | PR-AUC |
|---:|---:|---:|---:|
| 0.0668 | 0.8077 | 0.1235 | 0.6077 |

Recall increased from **48.90% to 80.77%**, but false positives increased from **4 to 2,052**. This resulted in a large drop in precision.

---

### 4.3 Hyperparameter Tuning

Four configurations were tested.

| Experiment | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Tuning 1 | 0.9706 | 0.3626 | 0.5280 | 0.6263 |
| Tuning 2 | 0.9545 | 0.4615 | 0.6222 | 0.6826 |
| Tuning 3 | 0.9468 | 0.4890 | 0.6449 | 0.6910 |
| **Tuning 4** | **0.9798** | **0.5330** | **0.6904** | **0.7020** |

Tuning 4 used:

```text
n_estimators = 200
max_depth = 6
learning_rate = 0.1
subsample = 0.9
colsample_bytree = 0.9
```

It produced the highest validation PR-AUC among the tested configurations.

---

## 5. Feature Importance

The five highest-ranked features were:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | Merchant State | 0.256868 |
| 2 | MCC | 0.154479 |
| 3 | Zip | 0.145451 |
| 4 | Merchant Name | 0.078494 |
| 5 | Year | 0.073029 |

The remaining features had lower individual importance.

---

## 6. Final Test Evaluation

The Tuning 4 configuration was trained on the training data and evaluated on the untouched test set.

| Metric | Test Result |
|---|---:|
| Precision | **0.9333** |
| Recall | **0.6154** |
| F1 | **0.7417** |
| PR-AUC | **0.7701** |

### Confusion Matrix

| | Predicted Legit | Predicted Fraud |
|---|---:|---:|
| Actual Legit | 149,805 | 8 |
| Actual Fraud | 70 | 112 |

The model detected **112 of 182 fraud cases** and produced **8 false positives**.

---

## 7. XGBoost Observations

- Class weighting increased recall but caused a large increase in false positives.
- Hyperparameter tuning improved validation PR-AUC from **0.6857** to **0.7020**.
- Tuning 4 gave the highest validation PR-AUC among the tested configurations.
- Training PR-AUC was **0.9523**, compared with **0.7020** on validation, showing a noticeable generalization gap.
- Merchant State, MCC and Zip had the highest feature-importance values.
- On the final test set, the model achieved **93.33% precision, 61.54% recall, 74.17% F1-score and 0.7701 PR-AUC**.

---

## 8. Files Included

The following files were used/generated for the XGBoost workflow:

### Scripts
```text
inspect_data.py
profile_data.py
prepare_dataset.py
split_dataset.py
preprocess_data.py
train_xgboost.py
tune_xgboost.py
check_overfitting.py
feature_importance.py
final_xgboost.py
generate_xgboost_graphs.py
outputs_person-2.py
```

### Dataset / Processed Data
```text
working_dataset.csv
X_train.csv
y_train.csv
X_validation.csv
y_validation.csv
X_test.csv
y_test.csv
```

### Results
```text
experiment_results.csv
xgboost_tuning_results.csv
xgboost_experiment_results_table.csv
xgboost_final_results_table.csv
xgboost_confusion_matrix_table.csv
xgboost_feature_importance_table.csv
xgboost_generalization_table.csv
output_person2_xgboost.csv
```

### Graphs
```text
01_xgboost_experiment_comparison.png
02_final_xgboost_confusion_matrix.png
03_xgboost_precision_vs_recall.png
04_xgboost_generalization_gap.png
xgboost_feature_importance.png
```

