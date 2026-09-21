"""Person 1 - Classical ML baseline (LogReg + RandomForest).
Run: uv run python 02_person1_baseline.py [--n-sample 200000]

Strategy (24M rows is too big for laptop RF):
- stream the 2.3GB CSV in chunks, reservoir-sample N rows uniformly
- keeps original ~0.12% fraud ratio (realistic test set)
- stratified train/val/test split, preprocessing fit on TRAIN only
- experiments: baseline, class_weight, RF hyperparams, threshold tuning
"""
import argparse
import os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve)

DATA_DIR = kagglehub.dataset_download("ealtman2019/credit-card-transactions")
MAIN = os.path.join(DATA_DIR, "credit_card_transactions-ibm_v2.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person1_outputs")
os.makedirs(OUT, exist_ok=True)

USECOLS = ["Year","Month","Day","Time","Amount","Use Chip","Merchant State",
           "Zip","MCC","Errors?","Is Fraud?"]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = (df["Is Fraud?"] == "Yes").astype(int)
    # Amount "$134.09" -> float
    df["Amount"] = df["Amount"].replace(r"[\$,]", "", regex=True).astype(float)
    # Time "06:21" -> hour
    df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour.fillna(12).astype(int)
    # Zip -> float, NaN -> -1
    df["Zip"] = pd.to_numeric(df["Zip"], errors="coerce").fillna(-1)
    # categoricals -> fill missing
    for c in ["Use Chip", "Merchant State", "Errors?"]:
        df[c] = df[c].fillna("Missing").astype(str)
    df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce").fillna(-1).astype(int)
    return df

def sample_stream(n_target: int, seed=42):
    """Uniform sample across whole file + disk cache for reuse."""
    cache = os.path.join(OUT, f"sample_{n_target}_{seed}.parquet")
    if os.path.exists(cache):
        print(f"loading cached sample: {cache}")
        return pd.read_parquet(cache)
    rng = np.random.default_rng(seed)
    import subprocess
    total = int(subprocess.check_output(["wc", "-l", MAIN]).split()[0]) - 1
    print(f"total rows in file: {total:,}")
    idx = np.sort(rng.choice(total, size=min(n_target, total), replace=False))
    print(f"sampling {len(idx):,} rows uniformly...")
    rows = []
    CH = 500_000
    skip = 0
    ptr = 0  # pointer into idx
    for chunk in pd.read_csv(MAIN, usecols=USECOLS, chunksize=CH):
        n = len(chunk)
        # vectorized: which sampled indices fall in [skip, skip+n)
        lo = np.searchsorted(idx, skip, side="left")
        hi = np.searchsorted(idx, skip + n, side="left")
        if hi > lo:
            local = idx[lo:hi] - skip
            rows.append(chunk.iloc[local])
        skip += n
        print(f"  scanned {skip:,}/{total:,} kept={sum(map(len, rows)):,}", end="\r")
    print()
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(cache, index=False)
    print(f"cached -> {cache}")
    return df

def build_preprocess():
    num = ["Amount", "Hour", "Zip", "MCC", "Year", "Month", "Day"]
    cat = ["Use Chip", "Merchant State", "Errors?"]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ]), num, cat

def evaluate(name, y_true, y_pred, y_prob):
    out = {
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(confusion_matrix(y_true, y_pred)[0, 0]),
        "fp": int(confusion_matrix(y_true, y_pred)[0, 1]),
        "fn": int(confusion_matrix(y_true, y_pred)[1, 0]),
        "tp": int(confusion_matrix(y_true, y_pred)[1, 1]),
    }
    print(f"{name}: P={out['precision']:.4f} R={out['recall']:.4f} "
          f"F1={out['f1']:.4f} PR-AUC={out['pr_auc']:.4f} "
          f"cm=[TN {out['tn']} FP {out['fp']} FN {out['fn']} TP {out['tp']}]")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=200000)
    a = ap.parse_args()

    df = clean(sample_stream(a.n_sample))
    print(f"sampled: {len(df):,} fraud={int(df.y.sum()):,} ({df.y.mean()*100:.4f}%)")

    X = df.drop(columns=["y", "Is Fraud?", "Time"])
    y = df["y"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=42, stratify=ytr)  # 60/20/20
    print(f"train {len(Xtr):,} val {len(Xva):,} test {len(Xte):,} "
          f"(fraud: {ytr.sum()}/{yva.sum()}/{yte.sum()})")

    pre, num, cat = build_preprocess()
    results = []

    # --- Logistic Regression: baseline vs balanced ---
    for cw, tag in [(None, "logreg_orig"), ("balanced", "logreg_balanced")]:
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(
            max_iter=1000, class_weight=cw, C=1.0, solver="lbfgs"))])
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xva)[:, 1]
        results.append(evaluate(tag, yva, (prob >= 0.5).astype(int), prob))

    # --- Random Forest: baseline vs balanced + depth variants ---
    rf_configs = [
        ("rf_orig_d10", 200, 10, None),
        ("rf_balanced_d10", 200, 10, "balanced"),
        ("rf_balanced_d20", 200, 20, "balanced"),
    ]
    best, best_auc = None, -1
    rf_probs = {}
    for tag, n_est, depth, cw in rf_configs:
        pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, class_weight=cw,
            n_jobs=-1, random_state=42))])
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xva)[:, 1]
        rf_probs[tag] = prob
        r = evaluate(tag, yva, (prob >= 0.5).astype(int), prob)
        results.append(r)
        if r["pr_auc"] > best_auc:
            best, best_auc = tag, r["pr_auc"]
        if tag == "rf_balanced_d20":
            best_pipe = pipe  # keep for importance + final test

    # --- Threshold tuning on best RF (validation) ---
    prob = rf_probs[best]
    prec, rec, th = precision_recall_curve(yva, prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    bi = int(np.argmax(f1s))
    bth = float(th[min(bi, len(th)-1)])
    print(f"best val threshold by F1: {bth:.3f} (F1={f1s[bi]:.4f} P={prec[bi]:.4f} R={rec[bi]:.4f})")
    results.append(evaluate(f"{best}_thr{bth:.2f}", yva, (prob >= bth).astype(int), prob))

    # --- Final test eval with best RF @ tuned threshold ---
    tprob = best_pipe.predict_proba(Xte)[:, 1]
    results.append(evaluate(f"{best}_TEST_thr{bth:.2f}", yte, (tprob >= bth).astype(int), tprob))

    # --- Feature importance (RF d20 balanced) ---
    try:
        feats = best_pipe.named_steps["pre"].get_feature_names_out()
        imp = best_pipe.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False).head(20)
        fi.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)
        plt.figure(figsize=(8, 6))
        plt.barh(fi.feature[::-1], fi.importance[::-1])
        plt.xlabel("importance"); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
        print("saved feature_importance.csv/png")
    except Exception as e:
        print("importance plot skipped:", e)

    # --- PR curve on test ---
    prec, rec, _ = precision_recall_curve(yte, tprob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Best RF - PR curve (TEST)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pr_curve_test.png"), dpi=150)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT, "person1_results.csv"), index=False)
    print(f"\nAll outputs -> {OUT}/person1_results.csv")
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()
