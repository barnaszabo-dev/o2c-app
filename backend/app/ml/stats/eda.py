import numpy as np, pandas as pd
from scipy.stats import gaussian_kde
from app.core.dataset import load_schema
from app.ml.stats.common import series, describe, clean, t, DIM_LABEL

DIMS = {"region": "d_region_code", "allocation_group": "d_allocation_group", "customer_category": "d_customer_category", "early": "d_early", "company": "Company"}
CATS = {**DIMS, "pn_group": "PNGroup", "doc_type": "Sales Document type", "distribution_channel": "Distribution Channel", "material_group_5": "Material group 5"}

def _group_stats(R, mid, col, neg, lo, hi):
    out = []
    for k, g in R.groupby(col, dropna=True):
        s = series(g, mid, neg)
        if len(s) < 10: continue
        d = describe(s); viol = None
        try:
            sm = s.sample(min(len(s), 2000), random_state=1).clip(lo, hi); grid = np.linspace(lo, hi, 40); dens = gaussian_kde(sm)(grid); viol = (dens / dens.max()).tolist()
        except Exception: pass
        out.append(dict(group=str(k), n=d["n"], mean=d["mean"], median=d["median"], q1=d["p25"], q3=d["p75"], whisker_low=d["whisker_low"], whisker_high=d["whisker_high"],
                        n_outliers=d["n_outlier_high"] + d["n_outlier_low"], violin=viol))
    return sorted(out, key=lambda x: -x["n"])

def run(R: pd.DataFrame, measure: str, neg: bool, bins: int, lang: str):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}
    s = series(R, measure, neg); d = describe(s)
    if s.empty: return dict(n_rows=len(R), measure=measure, empty=True)
    lo, hi = float(s.quantile(.01)), float(s.quantile(.99))
    if hi <= lo: hi = lo + 1
    inside = s[(s >= lo) & (s <= hi)]; counts, edges = np.histogram(inside, bins=bins, range=(lo, hi))
    groups = {k: _group_stats(R, measure, c, neg, lo, hi) for k, c in DIMS.items()}
    cats = {k: [dict(value=str(v), n=int(n), pct=n / len(R) * 100) for v, n in R[c].value_counts().head(12).items()] for k, c in CATS.items()}
    summary = []
    for m in sc["measures"]:
        x = series(R, m["id"], neg); dd = describe(x); summary.append(dict(id=m["id"], label=m["label"], kind=m["kind"], **{k: dd.get(k) for k in ("n", "median", "mean", "sd", "min", "p25", "p75", "max", "n_negative", "n_outlier_high")}))
    lv = pd.to_numeric(R["Loaded volume"], errors="coerce").dropna()
    lvd = describe(lv) if len(lv) else {}
    base = R.drop(columns=[c for c in R.columns if c.startswith(("d_", "h_"))])
    miss = sorted([dict(column=c, n_missing=int(R[c].isna().sum()), pct=R[c].isna().mean() * 100) for c in base.columns if R[c].isna().any()], key=lambda x: -x["n_missing"])[:14]
    ins = [t(lang, "size", n=f"{len(R):,}", d=f"{R['d_sales_document_base'].nunique():,}")]
    ratio = d["mean"] / d["median"] if d["median"] else None
    if ratio: ins.append(t(lang, "skew", m=lab[measure], mean=f"{d['mean']:.1f}", med=f"{d['median']:.1f}", ratio=f"{ratio:.1f}", pct=f"{d['n_outlier_high'] / d['n'] * 100:.1f}"))
    for dim, gs in groups.items():
        big = [g for g in gs if g["n"] >= 30]
        if len(big) >= 2:
            a, b = max(big, key=lambda g: g["median"]), min(big, key=lambda g: g["median"])
            ins.append(t(lang, "grp", dim=DIM_LABEL.get(lang, DIM_LABEL["en"])[dim], hi=a["group"], hv=f"{a['median']:.1f}", lo=b["group"], lv=f"{b['median']:.1f}", diff=f"{a['median'] - b['median']:.1f}"))
    ins.append(t(lang, "causal"))
    return clean(dict(n_rows=len(R), n_documents=int(R["d_sales_document_base"].nunique()), measure=measure, measure_label=lab[measure], stats=d,
        hist=dict(edges=edges.tolist(), counts=counts.tolist(), clipped_low=int((s < lo).sum()), clipped_high=int((s > hi).sum()), lo=lo, hi=hi),
        groups=groups, categorical=cats, summary=summary, loaded_volume=lvd,
        quality=dict(duplicate_sales_documents=int(R["Sales Document"].duplicated().sum()), duplicate_rows=int(base.duplicated().sum()), missing_top=miss),
        insights=ins))
