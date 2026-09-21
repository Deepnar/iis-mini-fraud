# Credit Card Fraud Detection — Team Workload & ML Experiment Plan

## 1. Project Objective

Develop an intelligent credit card fraud detection system using predictive analytics, with the primary focus on **model training, class-imbalance handling, experimentation, and comparative evaluation**.

The project will investigate how different machine-learning algorithms and imbalance-handling strategies affect fraud-detection performance.

---

## 2. Dataset

**Selected Dataset:** IBM Synthetic Credit Card Transactions / TabFormer dataset

The dataset is preferred because it contains large-scale, synthetic but realistic transaction data with meaningful transaction/customer/merchant-related attributes, avoiding the major inference limitation of PCA-anonymized datasets such as the ULB `V1–V28` dataset.

### Important assumption

Before final experimentation, the team must inspect:

- Total number of transactions
- Number and percentage of fraudulent transactions
- Number of legitimate transactions
- Available transaction/customer/card/merchant fields
- Missing values
- Categorical features
- Feature cardinality
- Whether fraud labels are sufficiently distributed for the planned experiments

**Do not artificially define “low/medium/high fraud” as a business label.**

Instead, if appropriate, manipulate the **training-set class ratio** to study the effect of class imbalance.

---

# 3. Core Technical Strategy

All three members must perform technical ML work.

The team will share a common data foundation, but each member will own a different experimental direction.

```text
                         IBM Dataset
                              |
                              v
                    Common Data Pipeline
                              |
                 +------------+------------+
                 |            |            |
                 v            v            v
             Member 1     Member 2     Member 3
             Classical     Boosting     Imbalance
                 ML           ML        Experiments
                 |            |            |
                 +------------+------------+
                              |
                              v
                     Unified Evaluation
                              |
                              v
                       Model Comparison
                              |
                              v
                    Best Final Configuration
```

---

# 4. Common Work — ALL MEMBERS

The following work is shared and must be understood by everyone.

### Data Understanding

- Inspect dataset structure
- Understand every usable feature
- Identify target/fraud label
- Analyze class distribution
- Identify missing/invalid values
- Identify categorical and numerical variables

### Preprocessing

- Data cleaning
- Encoding categorical variables where required
- Numerical preprocessing/scaling where required
- Feature engineering
- Train/validation/test splitting
- Prevent data leakage

### Evaluation Standard

The test set must remain untouched until final evaluation.

Primary metrics:

- Precision
- Recall
- F1-score
- PR-AUC
- Confusion Matrix

Accuracy will **not** be treated as the primary metric because fraud detection is an imbalanced classification problem.

---

# 5. Person 1 — Classical ML & Baseline Lead

## Role

**Classical Machine Learning Model Owner**

Person 1 already has comparatively stronger AI/ML knowledge, so this person will establish the baseline and develop strong conventional ML models.

## Technical Responsibilities

### Model Development

Implement and optimize:

1. Logistic Regression
2. Random Forest

### Experiments

Investigate:

- Baseline performance
- Class weighting
- Random Forest hyperparameters
- Feature importance
- Decision threshold tuning

### Suggested parameters to investigate

For Logistic Regression:

- `C`
- `class_weight`
- solver

For Random Forest:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `class_weight`
- `max_features`

### Technical Deliverables

- Working Logistic Regression implementation
- Working Random Forest implementation
- Baseline results
- Class-weight experiments
- Hyperparameter experiments
- Feature-importance analysis
- Threshold analysis
- Final results table for Person 1

### Key Question

> How effectively can classical machine-learning classifiers detect fraudulent transactions, and how much does class weighting improve detection?

---

# 6. Person 2 — Gradient Boosting & Advanced Model Lead

## Role

**Boosting / High-Performance Tabular ML Owner**

Person 2 will focus on advanced tree-based models that are highly suitable for structured/tabular data.

## Technical Responsibilities

### Model Development

Implement and compare:

1. XGBoost
2. LightGBM

If LightGBM creates unnecessary environment/setup problems, use **XGBoost as the primary advanced model** and do not waste project time on tooling issues.

### Experiments

Investigate:

- Baseline boosting performance
- Class imbalance weighting
- Hyperparameter optimization
- Feature importance
- Overfitting vs generalization

### XGBoost parameters to investigate

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `scale_pos_weight`
- Regularization parameters

### Technical Deliverables

- XGBoost implementation
- Optional LightGBM implementation
- Hyperparameter search
- Class-weight experiments
- Feature-importance analysis
- Validation results
- Final results table for Person 2

### Key Question

> Can gradient-boosting models outperform classical ML approaches for credit-card fraud detection?

---

# 7. Person 3 — Class Imbalance & Sampling Lead

## Role

**Imbalanced Learning / Experimental Design Owner**

Person 3 will investigate how the composition of the training data affects fraud-detection performance.

This is not simply “making fraud levels.” The experiment will study **training-set class ratios and imbalance-handling methods**.

## Technical Responsibilities

### Experiment A — Original Distribution

Train using the naturally occurring dataset distribution.

Example:

```text
Legitimate : Fraud
Original ratio
```

### Experiment B — Random Undersampling

Create controlled training distributions such as:

```text
100 : 1
50  : 1
20  : 1
10  : 1
```

The exact ratios must be chosen **after inspecting the actual IBM dataset distribution**.

Do not remove information from the test set.

### Experiment C — SMOTE

Apply SMOTE only to the training data.

Compare different minority oversampling levels where appropriate.

### Experiment D — Class Weighting

Compare model performance using different fraud-class weights.

### Technical Deliverables

- Original-distribution experiment
- Undersampling experiments
- SMOTE experiments
- Class-weight experiments
- Comparison of training ratios
- Precision/Recall/F1/PR-AUC analysis
- Final results table for Person 3

### Key Question

> How does the training-data class distribution and imbalance-handling strategy affect fraud-detection performance?

---

# 8. Why These Three Roles Were Chosen

The workload is deliberately divided into **three technically distinct ML dimensions**.

### Person 1

**Algorithm dimension — Classical ML**

```text
Logistic Regression
Random Forest
```

### Person 2

**Algorithm dimension — Boosting**

```text
XGBoost
LightGBM
```

### Person 3

**Data-distribution dimension — Imbalanced Learning**

```text
Original distribution
Undersampling
SMOTE
Class weighting
```

This avoids the common problem where one person does all the modeling while the others only prepare data or documentation.

---

# 9. Final Combined Experiment

After individual experiments, the team will combine results.

The comparison should answer two major questions:

## Question 1 — Which algorithm performs best?

Compare:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM (if feasible)
```

## Question 2 — Which imbalance strategy performs best?

Compare:

```text
Original distribution
Class weighting
Random undersampling
SMOTE
```

---

# 10. Final Evaluation

All final candidates must be evaluated on the **same untouched test set**.

### Primary metrics

| Metric | Importance |
|---|---|
| Precision | Very important |
| Recall | Very important |
| F1-score | Very important |
| PR-AUC | Primary overall comparison metric |
| Confusion Matrix | Required |
| Accuracy | Supporting metric only |

### Why?

A fraud model that predicts almost every transaction as legitimate can achieve extremely high accuracy while detecting almost no fraud.

Therefore:

> **High accuracy alone does not mean a good fraud-detection model.**

The final model should balance:

- Catching genuine fraud
- Avoiding excessive false alarms

---

# 11. Final Model Selection

The team will not select the model simply because it has the highest accuracy.

The final model will be selected using:

1. PR-AUC
2. Recall
3. Precision
4. F1-score
5. False-positive/false-negative trade-off
6. Stability across validation experiments

The team should explicitly document why the selected model is preferable.

---

# 12. Expected Final Comparison

The final report should contain a table similar to:

| Model | Imbalance Strategy | Precision | Recall | F1 | PR-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Original | — | — | — | — |
| Logistic Regression | Class Weight | — | — | — | — |
| Random Forest | Original | — | — | — | — |
| Random Forest | Undersampling | — | — | — | — |
| XGBoost | Class Weight | — | — | — | — |
| XGBoost | SMOTE | — | — | — | — |
| Best Model | Best Strategy | — | — | — | — |

Values are filled only after experiments are performed.

---

# 13. Technology Stack

## Core

- Python
- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost
- Optional: LightGBM

## Imbalanced Learning

- imbalanced-learn

## Visualization

- Matplotlib
- Seaborn

## Experimentation

- Jupyter Notebook or Python scripts

## Optional

- Optuna for automated hyperparameter optimization

**PyTorch is not required.**

The dataset is structured/tabular data, so tree-based models and conventional ML provide a strong and appropriate starting point.

---

# 14. Recommended Project Workflow

```text
PHASE 1
Dataset inspection
        |
        v
Feature understanding
        |
        v
Class distribution analysis
        |
        v
Common preprocessing
        |
        v

PHASE 2 — PARALLEL DEVELOPMENT

Person 1              Person 2              Person 3
Classical ML          Boosting ML            Imbalance ML
    |                     |                      |
    v                     v                      v
LR + RF              XGBoost + LGBM       SMOTE / Sampling /
    |                     |                Class Weighting
    +---------------------+----------------------+
                          |
                          v

PHASE 3
Unified validation
        |
        v
Model comparison
        |
        v
Threshold optimization
        |
        v

PHASE 4
Final untouched test evaluation
        |
        v
Final model selection
        |
        v
Results + conclusions
```

---

# 15. Important Rules

### Rule 1 — Never leak test data

Do not perform SMOTE, undersampling, scaling, feature selection, or other learned preprocessing using the complete dataset before splitting.

### Rule 2 — Keep the test distribution realistic

The test set should preserve the original fraud/legitimate distribution.

### Rule 3 — Do not invent fraud labels

“Low/medium/high fraud” should not be created unless there is a genuine business definition and dataset support.

The experiment should manipulate **training class ratios**, not redefine what fraud means.

### Rule 4 — Don't optimize for accuracy

Fraud detection requires attention to recall, precision, F1 and especially PR-AUC.

### Rule 5 — Don't force deep learning

PyTorch should only be introduced if experiments demonstrate a genuine reason to use neural networks.

### Rule 6 — Every member must produce experimental results

Each person must have:

- Code
- Trained models/experiments
- Evaluation metrics
- Graphs/tables
- Technical conclusions

---

# 16. Final Project Direction

The project is fundamentally a **comparative machine-learning study**, not a frontend/backend development project.

### Core research direction

> **Investigate how machine-learning algorithms and class-imbalance strategies influence the detection of fraudulent credit-card transactions.**

The final outcome is a technically justified answer to:

> **Which model + imbalance strategy provides the best fraud-detection performance on the selected dataset?**
