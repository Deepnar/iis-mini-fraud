# Person 2 — XGBoost Results

## Experiment Results

| experiment | precision_fraud | recall_fraud | f1_fraud | pr_auc | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline | 0.9570 | 0.4890 | 0.6473 | 0.6857 | 149808 | 4 | 93 | 89 |
| B_class_weighting | 0.0668 | 0.8077 | 0.1235 | 0.6077 | 147760 | 2052 | 35 | 147 |
| C_tuning_1 | 0.9706 | 0.3626 | 0.5280 | 0.6263 | 149810 | 2 | 116 | 66 |
| D_tuning_2 | 0.9545 | 0.4615 | 0.6222 | 0.6826 | 149808 | 4 | 98 | 84 |
| E_tuning_3 | 0.9468 | 0.4890 | 0.6449 | 0.6910 | 149807 | 5 | 93 | 89 |
| F_tuning_4 | 0.9798 | 0.5330 | 0.6904 | 0.7020 | 149810 | 2 | 85 | 97 |
| G_final_test | 0.9333 | 0.6154 | 0.7417 | 0.7701 | 149805 | 8 | 70 | 112 |

## Final XGBoost Test Result

| Metric | Value |
|---|---:|
| Precision | 0.9333 |
| Recall | 0.6154 |
| F1 Score | 0.7417 |
| PR-AUC | 0.7701 |

## Final Test Confusion Matrix

| Actual / Predicted | Legitimate | Fraud |
|---|---:|---:|
| **Legitimate** | 149805 | 8 |
| **Fraud** | 70 | 112 |

## Final Model Configuration

```text
n_estimators     = 200
max_depth        = 6
learning_rate    = 0.1
subsample        = 0.9
colsample_bytree = 0.9
random_state     = 42
```

## Notes

- Rows A–F are validation-set experiments.
- `G_final_test` is the final evaluation on the untouched test set.
- PR-AUC is the primary comparison metric for the project.
- The class-weighting experiment increased recall but produced substantially more false positives.
