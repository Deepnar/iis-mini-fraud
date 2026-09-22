"""Phase 2 (Person 3): Class Imbalance & Sampling experiments.
Rule: train/resample ONLY on train.csv; test.csv stays untouched.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, confusion_matrix,
                             precision_recall_curve)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

ROOT = r"C:\Users\Raman Tiwari\MyDevelopments\Credit_Example"
TARGET = "Is Fraud?"
SEED = 42

print("Loading train.csv / test.csv ...", flush=True)
train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test = pd.read_csv(os.path.join(ROOT, "test.csv"))
X_train_full = train.drop(columns=[TARGET])
y_train_full = train[TARGET].astype(int)
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET].astype(int)
print(f"train={X_train_full.shape} fraud={int(y_train_full.sum())} | "
      f"test={X_test.shape} fraud={int(y_test.sum())}", flush=True)

def make_rf(**kw):
    return RandomForestClassifier(n_estimators=100, random_state=SEED,
                                  n_jobs=-1, **kw)

experiments = {}  # name -> (X_fit, y_fit, rf_kwargs)

# A: baseline
experiments["A_baseline"] = (X_train_full, y_train_full, {})

# B: random undersampling to 20:1
print("Exp B: RandomUnderSampler(0.05) ...", flush=True)
rus = RandomUnderSampler(sampling_strategy=0.05, random_state=SEED)
Xb, yb = rus.fit_resample(X_train_full, y_train_full)
print(f"  resampled train: {Xb.shape}, fraud={int((yb == 1).sum())}, "
      f"legit={int((yb == 0).sum())}", flush=True)
experiments["B_undersample_0.05"] = (Xb, yb, {})

# C: SMOTE up to 10:1
print("Exp C: SMOTE(0.1) ... (this takes a few minutes)", flush=True)
sm = SMOTE(sampling_strategy=0.1, random_state=SEED)
Xc, yc = sm.fit_resample(X_train_full, y_train_full)
print(f"  resampled train: {Xc.shape}, fraud={int((yc == 1).sum())}, "
      f"legit={int((yc == 0).sum())}", flush=True)
experiments["C_smote_0.1"] = (Xc, yc, {})

# D: class weighting (raw data)
experiments["D_class_weight_balanced"] = (X_train_full, y_train_full,
                                          {"class_weight": "balanced"})

rows, curves = [], {}
for name, (Xf, yf, kw) in experiments.items():
    print(f"Training {name} ...", flush=True)
    clf = make_rf(**kw)
    clf.fit(Xf, yf)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    prec = precision_score(y_test, pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, pred, pos_label=1, zero_division=0)
    pr_auc = average_precision_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    rows.append({"experiment": name, "precision_fraud": round(prec, 4),
                 "recall_fraud": round(rec, 4), "f1_fraud": round(f1, 4),
                 "pr_auc": round(pr_auc, 4),
                 "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    p, r, _ = precision_recall_curve(y_test, proba)
    curves[name] = (r, p, pr_auc)
    print(f"  {name}: P={prec:.4f} R={rec:.4f} F1={f1:.4f} PR-AUC={pr_auc:.4f} "
          f"TN={tn} FP={fp} FN={fn} TP={tp}", flush=True)

results = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, "person3_results.csv")
results.to_csv(out_csv, index=False)
print("\n=== CONSOLIDATED RESULTS ===", flush=True)
print(results.to_string(index=False), flush=True)
print(f"Saved {out_csv}", flush=True)

# Combined PR-curve plot
plt.figure(figsize=(8, 6))
for name, (r, p, auc) in curves.items():
    plt.plot(r, p, label=f"{name} (AP={auc:.4f})")
no_skill = int(y_test.sum()) / len(y_test)
plt.axhline(no_skill, color="k", linestyle="--",
            label=f"Baseline (fraud rate={no_skill:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (evaluated on untouched test.csv)")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_png = os.path.join(ROOT, "pr_auc_curves.png")
plt.savefig(out_png, dpi=150)
print(f"Saved {out_png}", flush=True)
print("DONE", flush=True)
