"""Feature importance & model interpretation for the random forest: permutation importance (test period), impurity importance, SHAP (if installed),
per-category contributions, numeric dependence, a local explanation for one order and business-language insights generated from the results."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from app.core.dataset import load_schema
from app.ml.features.encoding import build
from app.ml.evaluation.metrics import metrics
from app.ml.models.forest import ForestModel
from app.ml.regression_service import _prep, select_features
from app.ml.stats.common import clean, TXT, t, _loc

TXT["en"].update(
 i_top="The most influential variables for the forest's predictions are {items} (permutation importance on the unseen test period: increase in MAE when the variable is shuffled).",
 i_share="{label} alone accounts for {pct}% of the total permutation importance.",
 i_agree="Permutation importance and SHAP rank the variables {how} (rank correlation {rho}).",
 i_cons="consistently", i_diff="rather differently",
 i_cat="Within {label}: {hi} pushes the predicted lead time up by {hv} on average, {lo} pulls it down by {lv} (relative to the average prediction).",
 i_num="Higher {label} values go with {dir} predicted lead times (rank correlation {rho}).", i_longer="longer", i_shorter="shorter",
 i_excl="{n} candidate variables are not used because they are not known at the prediction point or their timing is unconfirmed; their importance cannot be assessed here.",
 i_weak="The model explains little of the variation (test R²={r2}); importance describes what it relies on and stays modest in absolute terms.",
 i_corr="Variables that are strongly related to each other share importance, so a low value does not prove that a variable is irrelevant.",
 i_noshap="SHAP is not installed on the server (pip install shap); permutation and impurity importance are shown.",
 i_local="For order {doc}: the model predicts {pred} h (average prediction {base} h); actual {actual} h. Largest contributions: {items}.")
TXT["hu"].update(
 i_top="Az erdő előrejelzéseit leginkább befolyásoló változók: {items} (permutációs fontosság a nem látott teszt időszakon: az MAE növekedése, ha a változót összekeverjük).",
 i_share="A(z) {label} egyedül a teljes permutációs fontosság {pct}%-át adja.",
 i_agree="A permutációs fontosság és a SHAP {how} rangsorolja a változókat (rangkorreláció {rho}).",
 i_cons="egyezően", i_diff="eltérően",
 i_cat="A(z) {label} változón belül: {hi} átlagosan {hv}-val növeli az előre jelzett átfutási időt, {lo} {lv}-val csökkenti (az átlagos előrejelzéshez képest).",
 i_num="A(z) {label} magasabb értékei {dir} előre jelzett átfutási idővel járnak együtt (rangkorreláció {rho}).", i_longer="hosszabb", i_shorter="rövidebb",
 i_excl="{n} jelölt változó nincs használatban, mert nem ismert az előrejelzési ponton, vagy az időzítésük nem megerősített; a fontosságuk itt nem értékelhető.",
 i_weak="A modell a változékonyság csekély részét magyarázza (teszt R²={r2}); a fontosság azt írja le, mire támaszkodik a modell, és abszolút értékben szerény marad.",
 i_corr="Az egymással szorosan összefüggő változók megosztoznak a fontosságon, ezért az alacsony érték nem bizonyítja, hogy a változó lényegtelen.",
 i_noshap="A SHAP nincs telepítve a szerveren (pip install shap); a permutációs és a szennyezettség-alapú fontosság látható.",
 i_local="A(z) {doc} rendelésre a modell {pred} h-t jósol (átlagos előrejelzés {base} h); a tényleges érték {actual} h. Legnagyobb hozzájárulások: {items}.")

def _groups(meta, cols):
    g = {}
    for i, c in enumerate(cols): g.setdefault(meta[c]["feature"], []).append(i)
    return g

def _levels(Xrows, idx, cols, meta):
    """Category of each row reconstructed from its dummy columns (the reference category if none is set)."""
    names = [cols[i] for i in idx]; ref = meta[names[0]]["ref"]; M = Xrows[:, idx]; hit = M.argmax(axis=1); has = M.max(axis=1) > 0
    return np.array([meta[names[h]]["level"] if k else ref for h, k in zip(hit, has)])

def run_interpretation(R, req, neg):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}; target = req.measure; lang = req.lang
    nt, md, ml = max(10, min(req.n_trees, 500)), (None if req.max_depth in (None, 0) else max(2, min(req.max_depth, 40))), max(1, min(req.min_samples_leaf, 500))
    df, tr, te, cutoff = _prep(R, target, neg, req.test_fraction)
    cat, used, rejected, byid = select_features(df, tr, req, target)
    X, meta = build(df, used, tr); y = df["__y"]
    ok = X.notna().all(axis=1); df, X, y = df[ok], X[ok], y[ok]; tr, te = tr.intersection(df.index), te.intersection(df.index)
    log = req.log_target; g = (lambda v: np.log1p(v)) if log else (lambda v: v); ginv = (lambda v: np.expm1(v)) if log else (lambda v: v)
    cols = list(X.columns); Xtr, Xte, ytr, yte = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    rf = ForestModel(nt, md, ml).fit(Xtr.values, g(ytr).values); pte = ginv(rf.predict(Xte.values))
    mte, mb = metrics(yte, pte), metrics(yte, np.full(len(yte), float(ytr.median()))); mtr = metrics(ytr, ginv(rf.predict(Xtr.values)))
    groups = _groups(meta, cols); imp = rf.feature_importance(); rs = np.random.RandomState(0); base_mae = mte["mae"]; rows = {}
    for fid, idx in groups.items():
        incs = []
        for _ in range(5):
            Xp = Xte.values.copy(); Xp[:, idx] = Xp[rs.permutation(len(Xp))][:, idx]; incs.append(np.abs(yte.values - ginv(rf.predict(Xp))).mean() - base_mae)
        rows[fid] = dict(id=fid, label=byid[fid]["label"], type=byid[fid]["type"], review=byid[fid]["review"], perm_mean=float(np.mean(incs)), perm_sd=float(np.std(incs)), impurity=float(imp[idx].sum()), shap_mean_abs=None)
    shap_ok, shap_err, categories, numeric, base, sample_n = False, None, {}, {}, None, 0
    try:
        import shap
        samp = Xte.sample(min(len(Xte), 400), random_state=0); ex = shap.TreeExplainer(rf.m); sv = np.asarray(ex.shap_values(samp.values, check_additivity=False)); base = float(np.ravel(ex.expected_value)[0]); shap_ok = True; sample_n = len(samp)
        conv = (lambda v: (np.exp(v) - 1) * 100) if log else (lambda v: v)
        for fid, idx in groups.items():
            c = sv[:, idx].sum(axis=1); rows[fid]["shap_mean_abs"] = float(np.abs(c).mean())
            if byid[fid]["type"] == "categorical":
                lv = _levels(samp.values, idx, cols, meta); d = pd.DataFrame({"lv": lv, "c": c}); gg = d.groupby("lv")["c"].agg(["mean", "size"])
                categories[fid] = [dict(level=str(k), n=int(r["size"]), contribution=float(conv(r["mean"]) if log else r["mean"])) for k, r in gg.iterrows() if r["size"] >= 5]
                categories[fid].sort(key=lambda x: -x["contribution"])
            else:
                xv = samp.values[:, idx[0]]; k = np.random.RandomState(1).permutation(len(xv))[:300]; numeric[fid] = dict(points=[[float(xv[i]), float(c[i])] for i in k], rho=float(spearmanr(xv, c)[0]) if np.std(xv) > 0 and np.std(c) > 0 else None)
    except Exception as e:
        shap_err = str(e)[:160]
    imps = sorted(rows.values(), key=lambda r: -r["perm_mean"])
    rho = None
    if shap_ok and len(imps) >= 3: rho = float(spearmanr([r["perm_mean"] for r in imps], [r["shap_mean_abs"] for r in imps])[0])
    # ---- local explanation for one order ----
    local = None
    if req.order:
        q = str(req.order).strip(); m = df[df["Sales Document"].astype(str) == q]
        if m.empty: m = df[df["d_sales_document_base"].astype(str) == q.split("#")[0]]
        if shap_ok and not m.empty:
            i0 = m.index[0]; xr = X.loc[[i0]].values; sv1 = np.asarray(ex.shap_values(xr, check_additivity=False))[0]; items = []
            for fid, idx in groups.items():
                val = _levels(xr, idx, cols, meta)[0] if byid[fid]["type"] == "categorical" else float(xr[0, idx[0]])
                items.append(dict(label=byid[fid]["label"], value=str(val) if byid[fid]["type"] == "categorical" else round(val, 2), contribution=float(sv1[idx].sum())))
            items.sort(key=lambda d: -abs(d["contribution"])); pred = float(ginv(rf.predict(xr))[0])
            local = dict(document=str(df.loc[i0, "Sales Document"]), prediction=pred, base=float(ginv(base)), actual=float(y.loc[i0]), in_test=bool(i0 in set(te)), log_target=log, contributions=items[:10])
        elif not m.empty: local = dict(error="shap_unavailable")
        else: local = dict(error="not_found", document=q)
    # ---- insights from the actual results ----
    ins = []
    top = [r for r in imps if r["perm_mean"] > 0][:3]
    if top: ins.append(t(lang, "i_top", items="; ".join("%s (+%s h)" % (r["label"], _loc(lang, "%.1f" % r["perm_mean"])) for r in top)))
    tot = sum(max(0, r["perm_mean"]) for r in imps)
    if top and tot > 0: ins.append(t(lang, "i_share", label=top[0]["label"], pct="%.0f" % (top[0]["perm_mean"] / tot * 100)))
    if rho is not None: ins.append(t(lang, "i_agree", how=t(lang, "i_cons" if rho > .7 else "i_diff"), rho="%.2f" % rho))
    unit = "%" if log else " h"
    for r in imps:
        cs = categories.get(r["id"])
        if cs and len(cs) >= 2 and r["perm_mean"] > 0:
            hi, lo = cs[0], cs[-1]; ins.append(t(lang, "i_cat", label=r["label"], hi=hi["level"], hv=_loc(lang, "%.1f" % hi["contribution"]) + unit, lo=lo["level"], lv=_loc(lang, "%.1f" % abs(lo["contribution"])) + unit)); break
    for fid, nv in numeric.items():
        if nv["rho"] is not None and abs(nv["rho"]) > .3: ins.append(t(lang, "i_num", label=byid[fid]["label"], dir=t(lang, "i_longer" if nv["rho"] > 0 else "i_shorter"), rho="%.2f" % nv["rho"])); break
    ins.append(t(lang, "i_excl", n=sum(1 for c in cat if not c["allowed"])))
    if mte["r2"] is not None and mte["r2"] < .2: ins.append(t(lang, "i_weak", r2="%.2f" % mte["r2"]))
    if any(byid[u["id"]]["review"] for u in used): ins.append(t(lang, "f_review"))
    if not shap_ok: ins.append(t(lang, "i_noshap"))
    if local and "contributions" in local:
        ins.append(t(lang, "i_local", doc=local["document"], pred="%.0f" % local["prediction"], base="%.0f" % local["base"], actual="%.0f" % local["actual"], items="; ".join("%s = %s (%s%s)" % (c["label"], c["value"], "+" if c["contribution"] > 0 else "", _loc(lang, "%.1f" % c["contribution"])) for c in local["contributions"][:3])))
    ins += [t(lang, "i_corr"), t(lang, "f_imp_note"), t(lang, "causal")]
    coverage = [dict(id=c["id"], label=c["label"], status="used" if any(u["id"] == c["id"] for u in used) else "excluded",
                     reason=None if any(u["id"] == c["id"] for u in used) else (c["reason"] or next((r["reason"] for r in rejected if r["id"] == c["id"]), "not selected"))) for c in cat]
    return clean(dict(model="random_forest", target=target, target_label=lab[target], prediction_point=req.prediction_point, log_target=log, params=dict(n_trees=nt, max_depth=md, min_samples_leaf=ml),
        split=dict(n_train=len(tr), n_test=len(te), cutoff=str(cutoff)[:10]), test=mte, train=mtr, baseline_median=dict(value=float(ytr.median()), **mb),
        importance=imps, rank_agreement=rho, shap=dict(available=shap_ok, error=shap_err, sample=sample_n, base_value=None if base is None else float(ginv(base)), unit="%" if log else "h"),
        categories=categories, numeric=numeric, local=local, coverage=coverage, insights=ins))
