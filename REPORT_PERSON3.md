# Person 3 Report — Credit Card Fraud Detection (Phase 1 & Phase 2)

Owner: Person 3 — Class Imbalance & Sampling Lead

## 1. Objective
Build a reproducible preprocessing + imbalance-handling pipeline for credit card
fraud detection and compare 4 controlled sampling strategies using a single
fixed model, evaluated strictly on an untouched holdout set.

## 2. Data & Phase 1 Methodology
- Raw file: `credit_card_transactions-ibm_v2.csv` (~2.35 GB, 24,386,900 rows).
  **Never committed to git** (excluded via `.gitignore`).
- Memory-safe two-pass chunked sampling (`chunksize=200,000`, `random_state=42`),
  never loading the full file into RAM:
  - Pass 1 counted only `Is Fraud?`: total=24,386,900 | fraud=29,757 |
    legit=24,357,143 → natural fraud ratio ≈ **0.001220**.
  - Pass 2 drew exact per-class ordinal samples: **500,000 rows
    (610 fraud + 499,390 legit)**, preserving the natural ratio, then shuffled.
- Preprocessing (`phase1_preprocess.py`):
  - Target `Is Fraud?`: Yes→1, No→0.
  - `Amount`: stripped `$`/commas → float.
  - `Time` → `hour`, `minute`; `Year/Month/Day` → `month`, `day_of_week`;
    dropped raw `Year/Month/Day/Time`.
  - `User`, `Card` → int; `MCC` → numeric (NaN→mode).
  - `Use Chip`, `Merchant State`, `Errors?` → LabelEncoded (NaN→"MISSING").
  - `Merchant Name`, `Merchant City`, `Zip` → frequency-encoded (`*_freq`).
  - Zero NaNs remaining; final feature set = 14 columns + target.
- Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` →
  `train.csv` (400,000: 399,512 legit / 488 fraud) and `test.csv`
  (100,000: 99,878 legit / 122 fraud). Test set frozen for all evaluation.

## 3. Phase 2 Methodology (`phase2_experiments.py`)
- Features = all columns except `Is Fraud?`; target = `Is Fraud?` (0/1).
- **Golden rule: train/resample only on `train.csv`; `test.csv` untouched.**
- Fixed model: `RandomForestClassifier(n_estimators=100, random_state=42,
  n_jobs=-1)` in all 4 experiments:
  - **A (baseline):** raw train, no modification.
  - **B (undersampling):** `RandomUnderSampler(sampling_strategy=0.05,
    random_state=42)` → 10,248 rows (488 fraud / 9,760 legit ≈ 20:1).
  - **C (SMOTE):** `SMOTE(sampling_strategy=0.1, random_state=42)` →
    439,463 rows (39,951 fraud / 399,512 legit = 10:1).
  - **D (class weighting):** raw train + `class_weight='balanced'`.
- Metrics on untouched `test.csv`: Precision / Recall / F1 (fraud class),
  PR-AUC (`average_precision_score` on `predict_proba`), confusion matrix.

## 4. Results (from `person3_outputs/person3_results.csv`)
| experiment | precision_fraud | recall_fraud | f1_fraud | pr_auc | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|
| A_baseline | 1.0000 | 0.3770 | 0.5476 | 0.6976 | 99878 | 0 | 76 | 46 |
| B_undersample_0.05 | 0.2299 | 0.6803 | 0.3437 | 0.5542 | 99600 | 278 | 39 | 83 |
| C_smote_0.1 | 0.7975 | 0.5164 | 0.6269 | 0.6654 | 99862 | 16 | 59 | 63 |
| D_class_weight_balanced | 0.9828 | 0.4672 | 0.6333 | 0.7013 | 99877 | 1 | 65 | 57 |

PR curves: `person3_outputs/pr_auc_curves.png` (all 4 curves + no-skill
baseline at fraud rate 0.00122).

## 5. Technical Deductions
1. **D (class_weight='balanced') is the best overall trade-off**: highest F1
   (0.6333) and PR-AUC (0.7013) with a single false positive — recommended
   default when false alarms are costly.
2. **B maximises recall (0.68)** but precision collapses (0.23, 278 FPs);
   viable only where missing fraud is far costlier than analyst review load.
3. **C (SMOTE) is the runner-up** (F1=0.6269) and beats baseline recall
   (0.516 vs 0.377) at modest FP cost (16); capping synthesis at 10:1 avoided
   excessive noise while materially enriching the minority class.
4. **A is the most conservative** (precision 1.0, zero FPs) but catches only
   ~38% of fraud — unsuitable alone given the ~0.12% base rate.
5. Train-only resampling + frozen stratified test split kept every comparison
   leakage-free and PR-AUC-focused, which is the right criterion at this
   imbalance level (accuracy/ROC-AUC would mislead).

## 6. Reproducibility
- `phase1_preprocess.py` → `train.csv` / `test.csv` (script only; large CSVs
  git-ignored).
- `phase2_experiments.py` → `person3_outputs/person3_results.csv`,
  `person3_outputs/pr_auc_curves.png`.
- All randomness pinned (`random_state=42`); RF `n_jobs=-1`.
