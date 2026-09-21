"""Person 1 FULL experiments: LogReg sweep + RF grid + thresholds + final test.
Run: uv run python 03_person1_full.py [--n-sample 200000]
Uses cached parquet from 02 step, so instant start.
"""
import argparse, os
import numpy as np
import pandas as pd
import kagglehub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, precision_recall_curve)

DATA_DIR = kagglehub.dataset_download("ealtman2019/credit-card-transactions")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "person1_outputs")
os.makedirs(OUT, exist_ok=True)

def clean(df):
    df = df.copy()
    # support both raw (Is Fraud?) and cached-cleaned (y) frames
    if "y" not in df.columns:
        df["y"] = (df["Is Fraud?"] == "Yes").astype(int)
    # Amount may be "$80.00" string (raw cache) or float (cleaned) -> normalize always
    if df["Amount"].dtype == object or str(df["Amount"].dtype).startswith("str"):
        df["Amount"] = pd.to_numeric(df["Amount"].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce").fillna(0.0)
    else:
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    if "Hour" not in df.columns:
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour.fillna(12).astype(int)
    df["Zip"] = pd.to_numeric(df["Zip"], errors="coerce").fillna(-1)
    for c in ["Use Chip", "Merchant State", "Errors?"]:
        df[c] = df[c].fillna("Missing").astype(str)
    df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce").fillna(-1).astype(int)
    return df

def make_pre():
    num = ["Amount", "Hour", "Zip", "MCC", "Year", "Month", "Day"]
    cat = ["Use Chip", "Merchant State", "Errors?"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return pre

def ev(name, yt, yp, ypr):
    cm = confusion_matrix(yt, yp)
    d = {"model": name,
         "precision": precision_score(yt, yp, zero_division=0),
         "recall": recall_score(yt, yp, zero_division=0),
         "f1": f1_score(yt, yp, zero_division=0),
         "pr_auc": average_precision_score(yt, ypr),
         "tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
    print(f"{name}: P={d['precision']:.4f} R={d['recall']:.4f} F1={d['f1']:.4f} "
          f"PR-AUC={d['pr_auc']:.4f} cm=[{d['tn']} {d['fp']} {d['fn']} {d['tp']}]")
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=200000)
    a = ap.parse_args()
    cache = os.path.join(OUT, f"sample_{a.n_sample}_42.parquet")
    assert os.path.exists(cache), f"missing {cache}, run 02 first"
    df = clean(pd.read_parquet(cache))
    print(f"loaded {len(df):,} fraud={int(df.y.sum())} ({df.y.mean()*100:.4f}%)")

    feat_cols = ["Year","Month","Day","Hour","Amount","Use Chip","Merchant State","Zip","MCC","Errors?"]
    X, y = df[feat_cols], df["y"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, test_size=0.25, random_state=42, stratify=ytr)
    print(f"train {len(Xtr)} val {len(Xva)} test {len(Xte)}")

    results, val_probs, pipes = [], {}, {}

    # ===== A. Logistic Regression sweep: C x class_weight =====
    print("\n== A. LogReg C x class_weight (val @0.5) ==")
    for C in [0.01, 0.1, 1.0, 10.0]:
        for cw, tag in [(None, "orig"), ("balanced", "bal")]:
            name = f"logreg_C{C}_{tag}"
            p = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(
                C=C, class_weight=cw, solver="lbfgs", max_iter=1000))])
            p.fit(Xtr, ytr)
            pr = p.predict_proba(Xva)[:, 1]
            val_probs[name], pipes[name] = pr, p
            results.append(ev(name, yva, (pr >= 0.5).astype(int), pr))

    # CV check on two representative LogReg configs (3-fold on train, PR-AUC)
    print("\n-- LogReg 3-fold CV (train) PR-AUC --")
    for C, cw in [(1.0, None), (1.0, "balanced")]:
        p = Pipeline([("pre", make_pre()), ("clf", LogisticRegression(
            C=C, class_weight=cw, solver="lbfgs", max_iter=1000))])
        sc = cross_validate(p, Xtr, ytr, cv=StratifiedKFold(3, shuffle=True, random_state=42),
                            scoring="average_precision", n_jobs=-1)
        print(f"logreg C={C} {cw}: CV PR-AUC={sc['test_score'].mean():.4f} +- {sc['test_score'].std():.4f}")

    # ===== B. Random Forest grid =====
    print("\n== B. RF grid (val @0.5) ==")
    rf_grid = [
        ("rf_est200_d10_orig", 200, 10, 2, None, None),
        ("rf_est200_d10_bal", 200, 10, 2, "balanced", None),
        ("rf_est200_d20_bal", 200, 20, 2, "balanced", None),
        ("rf_est200_dNone_bal", 200, None, 2, "balanced", None),
        ("rf_est200_d20_split5_bal", 200, 20, 5, "balanced", None),
        ("rf_est200_d20_bal_sub", 200, 20, 2, "balanced_subsample", None),
        ("rf_est200_d20_bal_sqrt", 200, 20, 2, "balanced", "sqrt"),
        ("rf_est300_d20_bal", 300, 20, 2, "balanced", None),
    ]
    for name, n, d, mss, cw, mf in rf_grid:
        p = Pipeline([("pre", make_pre()), ("clf", RandomForestClassifier(
            n_estimators=n, max_depth=d, min_samples_split=mss,
            min_samples_leaf=1, class_weight=cw, max_features=mf,
            n_jobs=-1, random_state=42))])
        p.fit(Xtr, ytr)
        pr = p.predict_proba(Xva)[:, 1]
        val_probs[name], pipes[name] = pr, p
        results.append(ev(name, yva, (pr >= 0.5).astype(int), pr))

    # ===== C. pick best by PR-AUC, threshold tune on val =====
    res = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n== val ranking (PR-AUC) ==")
    print(res[["model","precision","recall","f1","pr_auc"]].head(10).to_string(index=False))
    best = res.iloc[0]["model"]
    print(f"\nBEST on val: {best}")
    bp = val_probs[best]
    prec, rec, th = precision_recall_curve(yva, bp)
    f1s = 2*prec*rec/(prec+rec+1e-12)
    bi = int(np.argmax(f1s))
    bth = float(th[min(bi, len(th)-1)])
    print(f"tuned threshold: {bth:.3f} F1={f1s[bi]:.4f} P={prec[bi]:.4f} R={rec[bi]:.4f}")
    results.append(ev(f"{best}_thr{bth:.2f}_VAL", yva, (bp >= bth).astype(int), bp))

    # ===== D. final locked test eval (best pipe, default 0.5 + tuned) =====
    tp = pipes[best].predict_proba(Xte)[:, 1]
    results.append(ev(f"{best}_TEST_t05", yte, (tp >= 0.5).astype(int), tp))
    results.append(ev(f"{best}_TEST_thr{bth:.2f}", yte, (tp >= bth).astype(int), tp))

    # ===== E. plots + tables =====
    full = pd.DataFrame(results)
    full.to_csv(os.path.join(OUT, "person1_full_results.csv"), index=False)

    # PR curves: best 4 models on val
    plt.figure()
    for m in res.head(4)["model"]:
        pr_, rc_, _ = precision_recall_curve(yva, val_probs[m])
        plt.plot(rc_, pr_, label=m)
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Person1 val PR curves")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "pr_curves_val.png"), dpi=150)

    # PR curve best on test
    plt.figure()
    pr_, rc_, _ = precision_recall_curve(yte, tp)
    plt.plot(rc_, pr_); plt.xlabel("recall"); plt.ylabel("precision")
    plt.title(f"{best} TEST PR-AUC={average_precision_score(yte,tp):.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pr_curve_test_best.png"), dpi=150)

    # feature importance from best RF pipe (if RF won) else deepest balanced RF
    rf_best = best if best.startswith("rf_") else "rf_est200_d20_bal"
    try:
        pipe = pipes[rf_best]
        feats = pipe.named_steps["pre"].get_feature_names_out()
        imp = pipe.named_steps["clf"].feature_importances_
        fi = pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)
        fi.to_csv(os.path.join(OUT, "feature_importance_full.csv"), index=False)
        plt.figure(figsize=(9, 6))
        plt.barh(fi.head(20).feature[::-1], fi.head(20).importance[::-1])
        plt.xlabel("importance"); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "feature_importance_full.png"), dpi=150)
        print(f"\nTop 10 features ({rf_best}):")
        print(fi.head(10).to_string(index=False))
    except Exception as e:
        print("importance skipped:", e)

    print(f"\nSaved -> {OUT}/person1_full_results.csv")
    print(full.to_string(index=False))

if __name__ == "__main__":
    main()
