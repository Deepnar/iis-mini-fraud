"""Phase 1: Common Preprocessing & Train/Test Split (memory-safe, stratified 500k sample)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SRC = r"C:\Users\Raman Tiwari\MyDevelopments\Credit_Example\credit_card_transactions-ibm_v2.csv"
OUT_DIR = r"C:\Users\Raman Tiwari\MyDevelopments\Credit_Example"
CHUNK = 200_000
TARGET_N = 500_000
SEED = 42
TARGET_COL = "Is Fraud?"

print("=== PASS 1: counting rows + fraud cases (usecols only) ===", flush=True)
n_total, n_fraud = 0, 0
for ch in pd.read_csv(SRC, usecols=[TARGET_COL], chunksize=CHUNK):
    n_total += len(ch)
    n_fraud += int((ch[TARGET_COL] == "Yes").sum())
n_legit = n_total - n_fraud
fraud_ratio = n_fraud / n_total
print(f"total={n_total:,} fraud={n_fraud:,} legit={n_legit:,} fraud_ratio={fraud_ratio:.6f}", flush=True)

n_fraud_s = int(round(fraud_ratio * TARGET_N))
n_legit_s = TARGET_N - n_fraud_s
print(f"sample targets -> fraud={n_fraud_s:,} legit={n_legit_s:,}", flush=True)

rng = np.random.default_rng(SEED)
fraud_keep = np.sort(rng.choice(n_fraud, n_fraud_s, replace=False))
legit_keep = np.sort(rng.choice(n_legit, n_legit_s, replace=False))

print("=== PASS 2: streaming stratified sample ===", flush=True)
parts, cf, cl, pf, pl = [], 0, 0, 0, 0
for ch in pd.read_csv(SRC, chunksize=CHUNK, dtype=str):
    is_f = (ch[TARGET_COL] == "Yes").to_numpy()
    f_pos = np.flatnonzero(is_f)          # row positions of frauds in chunk
    nf, nl = int(is_f.sum()), len(ch) - int(is_f.sum())
    take = []
    while pf < n_fraud_s and fraud_keep[pf] < cf + nf:   # fraud ordinal -> local -> row
        take.append(int(f_pos[fraud_keep[pf] - cf])); pf += 1
    l_pos = np.flatnonzero(~is_f)
    while pl < n_legit_s and legit_keep[pl] < cl + nl:
        take.append(int(l_pos[legit_keep[pl] - cl])); pl += 1
    if take:
        parts.append(ch.iloc[take])
    cf += nf; cl += nl
    if (pf + pl) % 100000 < len(take):
        print(f"  collected={pf+pl:,}/500000 (fraud kept={pf:,}, legit kept={pl:,})", flush=True)

sample = pd.concat(parts, ignore_index=True)
sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
print(f"sampled shape={sample.shape} fraud={(sample[TARGET_COL]=='Yes').sum():,}", flush=True)

print("=== PREPROCESSING ===", flush=True)
df = sample.copy()
df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)                       # 1. target -> binary
df["Amount"] = df["Amount"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)  # 2. amount
t = df["Time"].astype(str).str.extract(r"(?P<hour>\d{1,2}):(?P<minute>\d{2})").astype(int)    # 3a. time
df["hour"], df["minute"] = t["hour"], t["minute"]
for c in ["Year", "Month", "Day"]:                                           # 3b. date signals
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["month"] = df["Month"].astype(int)
df["day_of_week"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=df["Day"]),
                                   errors="coerce").dt.dayofweek.fillna(-1).astype(int)
df = df.drop(columns=["Year", "Month", "Day", "Time"])                       # 3c. drop raw date cols

for c in ["User", "Card"]:                                                   # ids -> numeric
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce").fillna(df["MCC"].mode()[0] if not df["MCC"].mode().empty else 0).astype(int)
for c in ["Use Chip", "Merchant State", "Errors?"]:                          # 4a. low-cardinality label-encode
    df[c] = df[c].fillna("MISSING").astype(str)
    df[c] = LabelEncoder().fit_transform(df[c])
for c in ["Merchant Name", "Merchant City", "Zip"]:                          # 4b. high-cardinality frequency-encode
    s = df[c].fillna("MISSING").astype(str)
    df[c + "_freq"] = s.map(s.value_counts()).astype(int)
    df = df.drop(columns=[c])
assert df.isna().sum().sum() == 0, "NaNs remain!"
print(f"processed shape={df.shape} cols={list(df.columns)}", flush=True)

print("=== TRAIN/TEST SPLIT (80/20, stratify, seed=42) ===", flush=True)
y = df[TARGET_COL]
train, test = train_test_split(df, test_size=0.2, stratify=y, random_state=SEED)
train.to_csv(f"{OUT_DIR}\\train.csv", index=False)
test.to_csv(f"{OUT_DIR}\\test.csv", index=False)

def report(name, d):
    f = int(d[TARGET_COL].sum()); t = len(d)
    print(f"{name}: total={t:,} | legit={t-f:,} | fraud={f:,} | fraud_rate={f/t:.6f}", flush=True)

report("train.csv", train)
report("test.csv ", test)
print("SAVED train.csv + test.csv", flush=True)
