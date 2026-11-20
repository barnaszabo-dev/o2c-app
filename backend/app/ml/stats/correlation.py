import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, kruskal
from app.core.dataset import load_schema
from app.ml.stats.common import clean, t, pfmt

CAT = {"region": ("d_region_code", "Region"), "company": ("Company", "Company"), "doc_type": ("Sales Document type", "Sales Document type"),
       "allocation_group": ("d_allocation_group", "Allocation group"), "customer_category": ("d_customer_category", "T1/T2/RS"), "pn_group": ("PNGroup", "PNGroup"),
       "distribution_channel": ("Distribution Channel", "Distribution Channel"), "material_group_5": ("Material group 5", "Material group 5"),
       "early": ("d_early", "Early / Non-early"), "sales_rep": ("Sales Rep", "Sales Rep"), "source_of_allocation": ("Source of Allocation", "Source of Allocation")}
DEFAULT_VARS = ["po_to_so_sap", "so_to_erp", "erp_to_sched_match", "sched_match_to_stat", "stat_to_z5", "z5_to_oc", "po_to_oc", "loaded_volume"]
COMPONENTS = {"po_to_oc": {"po_to_so_sap", "so_to_erp", "erp_to_sched_match", "sched_match_to_stat", "stat_to_z5", "z5_to_oc"}}

def _col(R, v, neg):
    s = pd.to_numeric(R["Loaded volume"] if v == "loaded_volume" else R[f"h_{v}"], errors="coerce")
    return s.where(s >= 0) if (neg and v != "loaded_volume") else s

def _pair(x, y):
    m = x.notna() & y.notna(); n = int(m.sum())
    if n < 8 or x[m].nunique() < 2 or y[m].nunique() < 2: return dict(n=n, pearson=None, p_pearson=None, spearman=None, p_spearman=None)
    r, p = pearsonr(x[m], y[m]); rho, ps = spearmanr(x[m], y[m])
    return dict(n=n, pearson=r, p_pearson=p, spearman=rho, p_spearman=ps)

def run(R, measure, neg, variables, pair, lang):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}; lab["loaded_volume"] = "Loaded volume"
    vs = [v for v in (variables or DEFAULT_VARS) if v in lab]; df = pd.DataFrame({v: _col(R, v, neg) for v in vs})
    cells = {}
    for i, a in enumerate(vs):
        for b in vs[i:]:
            cells[f"{a}|{b}"] = dict(n=int(df[a].notna().sum()), pearson=1.0, spearman=1.0, p_pearson=0, p_spearman=0) if a == b else _pair(df[a], df[b])
    y = _col(R, measure, neg); assoc = []
    for k, (col, label) in CAT.items():
        g = pd.DataFrame({"g": R[col].astype("string"), "y": y}).dropna(); sz = g.groupby("g")["y"].transform("size"); g = g[sz >= 5]
        if g["g"].nunique() < 2: continue
        grand = g["y"].mean(); ssb = g.groupby("g")["y"].agg(lambda s: len(s) * (s.mean() - grand) ** 2).sum(); sst = ((g["y"] - grand) ** 2).sum()
        eta2 = ssb / sst if sst else 0; h, p = kruskal(*[s.values for _, s in g.groupby("g")["y"]])
        assoc.append(dict(variable=k, label=label, n=len(g), n_groups=int(g["g"].nunique()), eta=eta2 ** .5, eta_sq=eta2, kruskal_h=h, p_value=p))
    assoc.sort(key=lambda a: -a["eta"])
    lv = _pair(_col(R, "loaded_volume", neg), y)
    px, py = (pair if pair and len(pair) == 2 and all(p in lab for p in pair) else ["erp_to_sched_match", "po_to_oc"])
    sx, sy = _col(R, px, neg), _col(R, py, neg); mm = sx.notna() & sy.notna(); pts = pd.DataFrame({"x": sx[mm], "y": sy[mm]})
    pts = pts.sample(min(len(pts), 1200), random_state=1)
    ins = []
    cand = [(a, b, c) for k, c in cells.items() for a, b in [k.split("|")] if a != b and c["spearman"] is not None and not (b in COMPONENTS.get(a, ()) or a in COMPONENTS.get(b, ()))]
    if cand:
        a, b, c = max(cand, key=lambda x: abs(x[2]["spearman"]))
        ins.append(t(lang, "pair", a=lab[a], b=lab[b], rho=f"{c['spearman']:.2f}", r=f"{c['pearson']:.2f}", n=f"{c['n']:,}"))
    for a, comps in COMPONENTS.items():
        have = [c for c in vs if c in comps]
        if a in vs and have: ins.append(t(lang, "struct", a=lab[a], b=" / ".join(lab[c] for c in have)))
    if assoc:
        a = assoc[0]; ins.append(t(lang, "assoc", m=lab[measure], v=a["label"], eta=f"{a['eta']:.2f}", pct=f"{a['eta_sq'] * 100:.1f}", p=pfmt(a["p_value"])))
        if a["variable"] == "early": ins.append(t(lang, "early_note"))
        if a["eta"] < .1: ins.append(t(lang, "weak"))
    ins += [t(lang, "bign"), t(lang, "causal")]
    return clean(dict(n_rows=len(R), measure=measure, variables=[dict(id=v, label=lab[v]) for v in vs], cells=cells, association=assoc, loaded_volume_vs_target=lv,
                      pair=dict(x=px, y=py, x_label=lab[px], y_label=lab[py], points=pts.values.tolist(), **_pair(sx, sy)), insights=ins,
                      encoding_note="Nominal variables are not ordinal-encoded: association with the target is measured by the correlation ratio (eta) on the category groups, with a Kruskal-Wallis test."))
