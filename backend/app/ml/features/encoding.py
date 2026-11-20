"""Design-matrix construction. Fitted on the TRAIN rows only (no information from the test period). Nominal variables -> one-hot with the most frequent train category as reference; rare levels -> 'Other'; missing -> 'Missing'."""
import numpy as np, pandas as pd

def build(df, feats, train_idx, min_count=30):
    X, meta = {}, {}
    for f in feats:
        col = f["column"]
        if f["type"] == "numeric":
            X[f["label"]] = pd.to_numeric(df[col], errors="coerce"); meta[f["label"]] = dict(feature=f["id"], label=f["label"], type="numeric"); continue
        s = df[col].astype("string").fillna("Missing"); cnt = s.loc[train_idx].value_counts()
        keep = set(cnt[cnt >= min_count].index); s = s.where(s.isin(keep), "Other")
        cnt = s.loc[train_idx].value_counts(); ref = cnt.idxmax()
        for lvl in cnt.index:
            if lvl == ref: continue
            nm = f"{f['label']}: {lvl}"; X[nm] = (s == lvl).astype(float); meta[nm] = dict(feature=f["id"], label=f["label"], type="dummy", level=str(lvl), ref=str(ref))
    return pd.DataFrame(X, index=df.index), meta
