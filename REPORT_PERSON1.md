# Credit Card Fraud Detection — Classical ML Baseline (Logistic Regression + Random Forest)

## 1. Objective

Build and rigorously evaluate classical machine-learning baselines for credit-card
fraud detection on the IBM synthetic transaction dataset, and answer:

> How effectively can classical classifiers detect fraud, and how much does
> class weighting improve detection?

Accuracy is explicitly rejected as a primary metric. With ~0.12% fraud, a model
predicting "all legitimate" scores 99.88% accuracy while catching zero fraud.
Primary metrics are Precision, Recall, F1, PR-AUC (average precision) and the
confusion matrix. PR-AUC is the primary model-selection metric.

## 2. Dataset

Source: `kagglehub.dataset_download("ealtman2019/credit-card-transactions")`
(the Kaggle slug is a username; the payload IS the IBM synthetic dataset).

Files:

| File | Size | Content |
|---|---|---|
| `credit_card_transactions-ibm_v2.csv` | 2.35 GB | 24,386,900 transactions, main training data |
| `sd254_cards.csv` | 0.5 MB | one row per card (brand, type, limit, dark-web flag) |
| `sd254_users.csv` | 0.2 MB | one row per person (age, income, FICO, location) |
| `User0_credit_card_transactions.csv` | 1.9 MB | single-user slice, same schema, for debugging |

### 2.1 Main transaction schema

| Column | Example | Notes |
|---|---|---|
| `User`, `Card` | 0, 0 | IDs, join keys to cards/users tables |
| `Year`, `Month`, `Day`, `Time` | 2002, 9, 1, `06:21` | range 1991–2020, all 12 months; `Hour` derived from `Time` |
| `Amount` | `$134.09` | string with `$`, range −500.0 to 12390.5; negatives are refunds/adjustments; stripped to float |
| `Use Chip` | `Swipe Transaction` | 3 values: Swipe 15.39M, Chip 6.29M, Online 2.71M |
| `Merchant Name/City/State`, `Zip` | `.../La Verne/CA/91750` | State cardinality 223 (noisy); 2,720,821 missing states, 2,878,135 missing zips |
| `MCC` | 5300 | merchant category code, 109 values; top: 5411 grocery, 5499 food, 5541 gas, 5812 restaurant, 5912 drugstore, 5300, 4829, 4784, 4121, 7538 |
| `Errors?` | NaN / `Bad CVV` | 23,998,469 missing (= no error); otherwise `Bad CVV, Bad Card Number, Bad PIN, Bad Expiration, Bad Zipcode, Insufficient Balance, Technical Glitch` and combos |
| `Is Fraud?` | `Yes` / `No` | target, mapped to 1/0 |

### 2.2 Class distribution (full chunked scan, no full load)

* Total: 24,386,900
* Fraud: 29,757 (0.1220%)
* Legitimate: 24,357,143
* Fraud by year is spread 1996–2019 (peaks 2010: 3835, 2016: 3579, 2008: 3710), so a uniform random sample preserves the time mix.

### 2.3 Why sampling

24M rows × one-hot state/MCC does not fit a laptop RandomForest workflow.
All modeling uses a uniform 200,000-row sample (245 fraud, 0.1225% — same ratio
as full data), cached to `person1_outputs/sample_200000_42.parquet` for
reproducibility. Split is stratified 60/20/20:

* train 120,000 (147 fraud), val 40,000 (49 fraud), test 40,000 (49 fraud)
* test set untouched until final evaluation; scalers/encoders fit on train only.

## 3. Method

### 3.1 Preprocessing (fit on train only)

* Numeric: `Amount, Hour, Zip, MCC, Year, Month, Day` → `StandardScaler`
* Categorical: `Use Chip, Merchant State, Errors?` → `OneHotEncoder(handle_unknown="ignore")`
* Pipeline: `ColumnTransformer` + classifier, so no leakage.

### 3.2 Logistic Regression — theory and setup

Model: `z = w·x + b`, `p = 1/(1+e^(−z))`. Linear boundary in feature space,
trained by minimizing log-loss plus L2 penalty `||w||²/C`.

* `C`: inverse regularization strength. Small C (0.01) forces tiny weights
  (underfit); large C (10) lets the model fit harder.
* `solver='lbfgs'`, `max_iter=1000`.
* `class_weight`: `None` = every row equal (fraud drowned 800:1);
  `'balanced'` = fraud rows weighted ~800× so the optimizer cares.

Sweep: C ∈ {0.01, 0.1, 1.0, 10.0} × {orig, balanced} = 8 fits, evaluated on val
at threshold 0.5, plus 3-fold stratified CV (train, PR-AUC) for C=1.0 orig/bal.

### 3.3 Random Forest — theory and setup

Ensemble of decision trees; each tree trained on a bootstrap sample with a
random feature subset at each split; majority/probability vote. Controls:

* `n_estimators`: number of trees (200 vs 300 tested).
* `max_depth`: max tree depth (10, 20, None). Shallow = regularized.
* `min_samples_split`: min samples to split a node (2 vs 5).
* `max_features`: features considered per split (None = all, `sqrt`).
* `class_weight`: `None` vs `balanced` vs `balanced_subsample`
  (the latter rebalances each bootstrap separately).

Grid (8 configs, val @0.5):

* est200_d10_orig / est200_d10_bal
* est200_d20_bal / est200_dNone_bal / est200_d20_split5_bal
* est200_d20_bal_sub / est200_d20_bal_sqrt / est300_d20_bal

### 3.4 Threshold tuning and final evaluation

Default threshold 0.5 assumes balanced classes and is wrong at 0.12% fraud.
Procedure: pick best val model by PR-AUC → scan `precision_recall_curve` on val
→ pick threshold maximizing F1 → evaluate once on locked test at both 0.5 and
tuned threshold. Report full confusion matrices.

## 4. Results

### 4.1 Logistic Regression (val @0.5)

| Model | P | R | F1 | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| C0.01_orig | 0.000 | 0.000 | 0.000 | 0.036 | 39951 | 0 | 49 | 0 |
| C0.01_bal | 0.007 | 0.837 | 0.014 | 0.053 | 34305 | 5646 | 8 | 41 |
| C0.1_orig | 0.000 | 0.000 | 0.000 | 0.157 | 39951 | 0 | 49 | 0 |
| C0.1_bal | 0.007 | 0.857 | 0.014 | 0.169 | 34040 | 5911 | 7 | 42 |
| C1.0_orig | 0.750 | 0.061 | 0.113 | 0.173 | 39950 | 1 | 46 | 3 |
| C1.0_bal | 0.007 | 0.837 | 0.014 | 0.202 | 33993 | 5958 | 8 | 41 |
| C10_orig | 0.875 | 0.143 | 0.246 | 0.169 | 39950 | 1 | 42 | 7 |
| C10_bal | 0.007 | 0.837 | 0.014 | 0.206 | 33987 | 5964 | 8 | 41 |

CV (train, PR-AUC): C=1.0 orig 0.151 ± 0.032, balanced 0.113 ± 0.006.

Reading: C=0.01/0.1 orig predict all-legit (R=0). Larger C helps slightly.
Balancing flips LogReg from blind (R 0.06) to vigilant (R 0.84) at the cost of
~6000 false positives and P≈0.007. Best LogReg PR-AUC is only 0.206 — the
linear boundary cannot capture fraud interactions.

### 4.2 Random Forest (val @0.5)

| Model | P | R | F1 | PR-AUC |
|---|---|---|---|---|
| est200_d10_orig | 1.000 | 0.408 | 0.580 | 0.494 |
| est200_d10_bal | 0.038 | 0.612 | 0.071 | 0.198 |
| est200_d20_bal | 0.080 | 0.469 | 0.136 | 0.306 |
| est200_dNone_bal | 0.375 | 0.367 | 0.371 | 0.341 |
| est200_d20_split5_bal | 0.076 | 0.469 | 0.131 | 0.333 |
| est200_d20_bal_sub | 0.124 | 0.449 | 0.194 | 0.297 |
| est200_d20_bal_sqrt | 0.233 | 0.408 | 0.296 | 0.337 |
| est300_d20_bal | 0.079 | 0.490 | 0.137 | 0.306 |

Reading: with only 147 train frauds, shallow + original distribution wins
(PR-AUC 0.494 vs next 0.341). Balancing a shallow forest adds noise (0.49→0.19).
Depth helps balanced models (d10 0.19 → dNone 0.34) but never beats d10-orig.
`sqrt` features help precision (0.08→0.23). 300 trees = same as 200 (0.306),
so 200 is sufficient.

### 4.3 Threshold tuning (best val model: est200_d10_orig)

Val scan: best F1 at threshold 0.245 (F1 0.590, P 0.793, R 0.469).

| Eval | P | R | F1 | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| val @0.5 | 1.000 | 0.408 | 0.580 | 0.494 | 39951 | 0 | 29 | 20 |
| val @0.24 | 0.793 | 0.469 | 0.590 | 0.494 | 39945 | 6 | 26 | 23 |
| TEST @0.5 | 0.955 | 0.429 | 0.592 | 0.618 | 39950 | 1 | 28 | 21 |
| TEST @0.24 | 0.784 | 0.592 | 0.674 | 0.618 | 39943 | 8 | 20 | 29 |

Same model, no retraining: recall 0.43→0.59 (+8 fraud caught), F1 0.59→0.67,
at the cost of 1→8 false positives. This is the single cheapest win.

### 4.4 Feature importance (best RF, top 10)

| Feature | Importance |
|---|---|
| num__MCC | 0.252 |
| Merchant State Italy | 0.233 |
| num__Year | 0.097 |
| num__Amount | 0.086 |
| Merchant State Missing | 0.067 |
| Use Chip Online Transaction | 0.064 |
| num__Zip | 0.041 |
| num__Hour | 0.035 |
| Merchant State Algeria | 0.027 |
| Merchant State Nigeria | 0.022 |

Interpretation: fraud concentrates in specific MCCs, foreign/missing states,
online channel, plus amount/year effects. `Errors?` categories also contribute
further down the list. This matches domain intuition: card-not-present +
cross-border + error-prone transactions are riskiest.

### 4.5 Final model

* Config: `RandomForest(n_estimators=200, max_depth=10, class_weight=None)`,
  preprocessing as in §3.1, threshold **0.24**.
* Locked TEST: **P 0.784, R 0.592, F1 0.674, PR-AUC 0.618** (29/49 fraud caught, 8 FP).
* Artifact `person1_outputs/best_rf_model.joblib` (2.1 MB) is the same config
  retrained on train+val (160k rows) for deployment; on the same test it scores
  P 1.0 / R 0.41 @0.24 (threshold should be re-tuned after adding more data).
* Plots: `pr_curves_val.png`, `pr_curve_test_best.png`,
  `feature_importance_full.png`. Tables: `person1_full_results.csv`.

## 5. Conclusions for the team comparison

1. Classical ML works: RF PR-AUC 0.62 vs LogReg 0.21 on the same sample.
2. Class weighting is not universally good: it rescues LogReg recall
   (0.06→0.84) but wrecks RF ranking (0.49→0.19 at d10). Report both.
3. Threshold tuning is mandatory at 0.12% fraud: +0.08 F1 free.
4. RF beats LogReg because fraud is interaction-driven (channel × geography ×
   MCC), which a linear model cannot express.
5. Limitation: 200k sample holds only 245 frauds; confidence intervals are
   wide (±0.03 CV). Scale winning configs to 1M+ rows before the joint
   Person 1/2/3 final table.

## 6. Reproduce (Linux + uv)

```bash
uv python pin 3.12
uv sync
uv run python download_dataset.py
uv run python 01_inspect.py
uv run python 02_person1_baseline.py --n-sample 200000
uv run python 03_person1_full.py --n-sample 200000
```

Outputs land in `person1_outputs/`. Re-running `03` is instant (uses cached parquet).

## 7. Files in this repo

* `download_dataset.py` — kagglehub download
* `01_inspect.py` — chunked full-file inspection (24M rows, no RAM blowup)
* `02_person1_baseline.py` — uniform sampling + cache + LogReg/RF baselines + threshold
* `03_person1_full.py` — LogReg C sweep + CV, RF 8-config grid, threshold tune, test eval, plots
* `person1_outputs/` — results CSVs, PR curves, feature importance, cached sample, `best_rf_model.joblib`
* `Credit_Card_Fraud_Team_ML_Plan.md` — team plan (Person 1 = Classical ML Lead)
