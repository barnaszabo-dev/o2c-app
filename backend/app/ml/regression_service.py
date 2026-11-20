"""Linear and multiple linear regression (statsmodels OLS, HC3 robust errors). Leakage-guarded features, chronological+grouped validation."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats as sps
from sklearn.model_selection import GroupKFold
from fastapi import HTTPException
from app.core.dataset import load_schema
from app.ml.features.catalog import catalog
from app.ml.features.encoding import build
from app.ml.evaluation.splits import chrono_group_split
from app.ml.evaluation.metrics import metrics
from app.ml.models.ols import fit_ols, predict, vif
from app.ml.stats.common import clean, TXT, t, pfmt, _loc

TXT["en"].update(
 r_fit="On unseen later orders ({nte} items from {cutoff} on) the model reaches R²={r2}, MAE={mae} h and RMSE={rmse} h. Always predicting the training median gives MAE={bmae} h, so the model {verdict}.",
 r_better="reduces the MAE by {pct}%", r_worse="does not beat this naive baseline",
 r_cv="Grouped cross-validation on the training period: R²={r2} ± {sd}, MAE={mae} h (orders never split between folds).",
 r_pos="Largest positive associations with longer {y}: {items}.", r_neg="Largest negative associations (shorter {y}): {items}.",
 r_vif="Multicollinearity: {items} have VIF above 10, so their individual coefficients are unstable.",
 r_block="{n} candidate variables were excluded by the leakage guard because they are not known at the prediction point ({pt}) or their timing is unconfirmed.",
 r_log="The target is log-transformed: effects are relative (percent) changes, and R²/MAE/RMSE are reported back in hours.",
 r_lin="Each additional unit of {x} is associated with a {b} h change in {y} (slope p={p}); in the training period R²={r2}.",
 r_leak="Warning: {x} is only known after {y} has ended, so it cannot be used to predict it in advance; read this as a descriptive relationship only.",
 r_struct="Warning: {x} is a component of {y}; a relationship is expected by construction.",
 r_low="Low explanatory power: most of the variation in {y} is not captured by these variables.")
TXT["hu"].update(
 r_fit="A később leadott, a modell által nem látott rendeléseken ({nte} tétel, {cutoff} naptól) a modell R²={r2}, MAE={mae} h és RMSE={rmse} h értéket ér el. Ha mindig a tanítóidőszak mediánját becsüljük, az MAE={bmae} h, így a modell {verdict}.",
 r_better="{pct}%-kal csökkenti az MAE-t", r_worse="nem veri meg ezt az egyszerű alapmódszert",
 r_cv="Csoportos keresztvalidáció a tanítóidőszakon: R²={r2} ± {sd}, MAE={mae} h (egy rendelés tételei sosem kerülnek különböző foldokba).",
 r_pos="A(z) {y} hosszabb idejével leginkább pozitívan összefüggő tényezők: {items}.", r_neg="A legerősebb negatív összefüggések (rövidebb {y}): {items}.",
 r_vif="Multikollinearitás: {items} VIF-értéke 10 fölött van, ezért az egyedi együtthatóik instabilak.",
 r_block="{n} jelölt változót az adatszivárgás-védelem kizárt, mert nem ismert az előrejelzési ponton ({pt}), vagy az időzítésük nem megerősített.",
 r_log="A célváltozó logaritmizált: a hatások relatív (százalékos) változások, az R²/MAE/RMSE órákra visszaszámolva szerepel.",
 r_lin="Minden további egység a(z) {x} értékében {b} h változással jár együtt a(z) {y} esetén (meredekség p={p}); a tanítóidőszakon R²={r2}.",
 r_leak="Figyelem: a(z) {x} csak a(z) {y} lezárulása után ismert, ezért előre jelezni nem használható; csak leíró összefüggésként értelmezd.",
 r_struct="Figyelem: a(z) {x} a(z) {y} része, ezért az összefüggés szerkezetileg elvárt.",
 r_low="Alacsony magyarázóerő: a(z) {y} változékonyságának nagy részét ezek a változók nem fogják meg.")

def _prep(R, target, neg, tf):
    df = R.copy(); y = pd.to_numeric(df[f"h_{target}"], errors="coerce"); ok = y.notna() & (y >= 0 if neg else True); df = df[ok].copy()
    df["__y"] = y[ok]; po = pd.to_datetime(df["Purchase Order Date"]); df["__po_weekday"] = po.dt.day_name(); df["__po_hour"] = po.dt.hour.astype(float)
    if len(df) < 100: raise HTTPException(422, f"Only {len(df)} usable observations (minimum 100 for regression)")
    tr, te, cutoff = chrono_group_split(df, tf)
    if len(te) < 30: raise HTTPException(422, "Test period too small")
    return df, tr, te, cutoff

def _hist(v, bins=30):
    v = np.asarray(v, float); lo, hi = np.quantile(v, [.01, .99]); c, e = np.histogram(v[(v >= lo) & (v <= hi)], bins=bins, range=(lo, hi)); return dict(edges=e.tolist(), counts=c.tolist(), lo=lo, hi=hi)

def _sample(a, b, n=800, seed=1):
    idx = np.random.RandomState(seed).permutation(len(a))[:n]; return [[float(a[i]), float(b[i])] for i in idx]

def _fit_eval(df, tr, te, X, y, log, lang):
    Xtr, Xte = X.loc[tr], X.loc[te]; ytr, yte = y.loc[tr], y.loc[te]
    g = (lambda v: np.log1p(v)) if log else (lambda v: v); ginv = (lambda v: np.expm1(v)) if log else (lambda v: v)
    m = fit_ols(Xtr, g(ytr)); ptr = ginv(predict(m, Xtr)); pte = ginv(predict(m, Xte))
    base = float(ytr.median()); mt = metrics(yte, pte, Xtr.shape[1]); mtr = metrics(ytr, ptr, Xtr.shape[1])
    mtr["adj_r2"] = float(m.rsquared_adj) if not log else mtr["adj_r2"]
    mb = metrics(yte, np.full(len(yte), base))
    cv = []; grp = df.loc[tr, "d_sales_document_base"].values
    for a, b in GroupKFold(5).split(Xtr, groups=grp):
        mm = fit_ols(Xtr.iloc[a], g(ytr.iloc[a]), robust=False); cv.append(metrics(ytr.iloc[b], ginv(predict(mm, Xtr.iloc[b]))))
    cvs = dict(folds=cv, r2_mean=np.mean([c["r2"] for c in cv]), r2_sd=np.std([c["r2"] for c in cv]), mae_mean=np.mean([c["mae"] for c in cv]), rmse_mean=np.mean([c["rmse"] for c in cv]))
    res = np.asarray(yte) - np.asarray(pte)
    return m, dict(test=mt, train=mtr, baseline_median=dict(value=base, **mb), cv=cvs, residual_hist=_hist(res), actual_vs_pred=_sample(np.asarray(yte), np.asarray(pte)),
                   resid_vs_fitted=_sample(np.asarray(pte), res)), pte

def _fit_insight(r, cutoff, nte, lang, log):
    te, b = r["test"], r["baseline_median"]; better = b["mae"] and te["mae"] < b["mae"]
    verdict = t(lang, "r_better", pct=f"{(1 - te['mae'] / b['mae']) * 100:.0f}") if better else t(lang, "r_worse")
    out = [t(lang, "r_fit", nte=f"{nte:,}", cutoff=str(cutoff)[:10], r2=f"{te['r2']:.2f}", mae=f"{te['mae']:.1f}", rmse=f"{te['rmse']:.1f}", bmae=f"{b['mae']:.1f}", verdict=verdict),
           t(lang, "r_cv", r2=f"{r['cv']['r2_mean']:.2f}", sd=f"{r['cv']['r2_sd']:.2f}", mae=f"{r['cv']['mae_mean']:.1f}")]
    return out

def _drop_nested(df, tr, used):
    """Nominal variables that fully determine each other (e.g. Sales Rep -> Region, Source of Allocation -> Allocation group) are perfectly collinear.
    Keep the variable that comes first in the catalog (business-interpretable first), drop the other and say why."""
    cats = [u for u in used if u["type"] == "categorical"]; drop = {}
    S = {u["id"]: df.loc[tr, u["column"]].astype("string").fillna("Missing") for u in cats}
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            if a["id"] in drop or b["id"] in drop: continue
            g = pd.DataFrame({"a": S[a["id"]], "b": S[b["id"]]})
            if g.groupby("a")["b"].nunique().max() <= 1 or g.groupby("b")["a"].nunique().max() <= 1:
                drop[b["id"]] = f"perfectly collinear with {a['label']} (one determines the other); {a['label']} kept"
    return [u for u in used if u["id"] not in drop], [dict(id=k, label=next(u["label"] for u in used if u["id"] == k), reason=v) for k, v in drop.items()]

def select_features(df, tr, req, target):
    """Shared by all models: leakage-guarded, non-constant, non-nested variables (+ the reasons for everything that was left out)."""
    cat = catalog(req.prediction_point, req.include_review, target); ids = req.features or [c["id"] for c in cat if c["allowed"]]
    byid = {c["id"]: c for c in cat}; used, rejected = [], []
    for i in ids:
        c = byid.get(i)
        if not c: continue
        if not c["allowed"]: rejected.append(dict(id=i, label=c["label"], reason=c["reason"])); continue
        if df[c["column"]].nunique(dropna=True) < 2: rejected.append(dict(id=i, label=c["label"], reason="constant in this selection")); continue
        used.append(c)
    used, nested = _drop_nested(df, tr, used); rejected += nested
    if not used: raise HTTPException(422, "No usable explanatory variables for this prediction point/selection")
    return cat, used, rejected, byid

def run_multiple(R, req, neg):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}; target = req.measure
    df, tr, te, cutoff = _prep(R, target, neg, req.test_fraction)
    cat, used, rejected, byid = select_features(df, tr, req, target)
    X, meta = build(df, used, tr); y = df["__y"]
    ok = X.notna().all(axis=1); dropped = int((~ok).sum()); df, X, y = df[ok], X[ok], y[ok]; tr, te = tr.intersection(df.index), te.intersection(df.index)
    log = req.log_target; m, ev, pte = _fit_eval(df, tr, te, X, y, log, req.lang)
    P, SE, PV, CI = (np.asarray(v) for v in (m.params, m.bse, m.pvalues, m.conf_int())); coefs = []
    vf = vif(X.loc[tr]); fvif = {}
    for nm, mt in meta.items(): fvif[mt["feature"]] = max(fvif.get(mt["feature"], 0), vf[nm])
    sd = X.loc[tr].std()
    for i, nm in enumerate(["const"] + list(X.columns)):
        b = float(P[i]); mt = meta.get(nm, dict(type="intercept", label="Intercept"))
        eff = (np.exp(b) - 1) * 100 if log else b
        coefs.append(dict(name=nm, label=mt.get("label"), type=mt["type"], level=mt.get("level"), ref=mt.get("ref"), coef=b, effect=eff, effect_unit="%" if log else "h", se=float(SE[i]),
                          p=float(PV[i]), ci_low=float(CI[i][0]), ci_high=float(CI[i][1]), std_beta=None if nm == "const" or not sd.get(nm) else b * sd[nm] / (y.loc[tr].std() if not log else np.log1p(y.loc[tr]).std()),
                          vif=None if nm == "const" else vf[nm]))
    lang = req.lang; ins = _fit_insight(ev, cutoff, len(te), lang, log)
    sig = [c for c in coefs if c["type"] != "intercept" and c["p"] < .05]; unit = "%" if log else " h"
    def fmt(c):
        lvl = (": %s vs %s" % (c["level"], c["ref"])) if c["type"] == "dummy" else ""
        return "%s%s (%s%s%s)" % (c["label"], lvl, "+" if c["effect"] > 0 else "", _loc(lang, "%.1f" % c["effect"]), unit)
    pos = sorted([c for c in sig if c["effect"] > 0], key=lambda c: -abs(c["effect"]))[:3]; ng = sorted([c for c in sig if c["effect"] < 0], key=lambda c: -abs(c["effect"]))[:3]
    if pos: ins.append(t(lang, "r_pos", y=lab[target], items="; ".join(fmt(c) for c in pos)))
    if ng: ins.append(t(lang, "r_neg", y=lab[target], items="; ".join(fmt(c) for c in ng)))
    hv = [k for k, v in fvif.items() if v > 10]
    if hv: ins.append(t(lang, "r_vif", items=", ".join(byid[k]["label"] for k in hv)))
    if ev["test"]["r2"] is not None and ev["test"]["r2"] < .2: ins.append(t(lang, "r_low", y=lab[target]))
    nb = sum(1 for c in cat if not c["allowed"]); ins.append(t(lang, "r_block", n=nb, pt=req.prediction_point))
    if log: ins.append(t(lang, "r_log"))
    ins.append(t(lang, "causal"))
    return clean(dict(model="multiple_linear", target=target, target_label=lab[target], prediction_point=req.prediction_point, log_target=log, n_features=X.shape[1],
        split=dict(n_train=len(tr), n_test=len(te), cutoff=str(cutoff)[:10], rule="chronological hold-out by Purchase Order Date; all items of an order on the same side", dropped_rows_missing_numeric=dropped),
        used_features=[dict(id=u["id"], label=u["label"], type=u["type"], vif=fvif.get(u["id"])) for u in used], rejected=rejected,
        coefficients=coefs, f_pvalue=float(m.f_pvalue), **ev, insights=ins, robust_se="HC3"))

def run_linear(R, req, neg):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}; lab["loaded_volume"] = "Loaded volume (actual)"; ev_ = sc["events"]; target = req.measure; x = req.x or "so_to_erp"
    if x not in lab or x == target: raise HTTPException(404, "unknown or identical explanatory variable")
    df, tr, te, cutoff = _prep(R, target, neg, req.test_fraction)
    xs = pd.to_numeric(df["Loaded volume"] if x == "loaded_volume" else df[f"h_{x}"], errors="coerce")
    if neg and x != "loaded_volume": xs = xs.where(xs >= 0)
    ok = xs.notna(); df, xs = df[ok], xs[ok]; tr, te = tr.intersection(df.index), te.intersection(df.index)
    if len(tr) < 50 or len(te) < 30: raise HTTPException(422, "Too few observations with a valid explanatory value")
    X = pd.DataFrame({lab[x]: xs}); y = df["__y"]; log = req.log_target
    m, ev, pte = _fit_eval(df, tr, te, X, y, log, req.lang)
    P, SE, PV, CI = (np.asarray(v) for v in (m.params, m.bse, m.pvalues, m.conf_int())); a, b = float(P[0]), float(P[1]); xt = xs.loc[tr]; x0, x1 = np.quantile(xt, [.01, .99]); g = (lambda v: np.expm1(v)) if log else (lambda v: v)
    xe = sc["measures"]; xend = "real_loading" if x == "loaded_volume" else next(mm for mm in xe if mm["id"] == x)["end_event"]
    tm = next(mm for mm in xe if mm["id"] == target); lang = req.lang; ins = _fit_insight(ev, cutoff, len(te), lang, log)
    r = float(np.corrcoef(xt, y.loc[tr])[0, 1]); pr = float(PV[1])
    ins.insert(0, t(lang, "r_lin", x=lab[x], y=lab[target], b=f"{b:.2f}" if not log else f"{(np.exp(b) - 1) * 100:.2f} %", p=pfmt(pr), r2=f"{ev['train']['r2']:.2f}"))
    if xend and tm["end_event"] and ev_.index(xend) >= ev_.index(tm["end_event"]): ins.append(t(lang, "r_leak", x=lab[x], y=lab[target]))
    if x != "loaded_volume":
        xm = next(mm for mm in xe if mm["id"] == x)
        if xm["start_event"] and tm["start_event"] and ev_.index(xm["start_event"]) >= ev_.index(tm["start_event"]) and xm["end_event"] and tm["end_event"] and ev_.index(xm["end_event"]) <= ev_.index(tm["end_event"]): ins.append(t(lang, "r_struct", x=lab[x], y=lab[target]))
    if ev["test"]["r2"] is not None and ev["test"]["r2"] < .2: ins.append(t(lang, "r_low", y=lab[target]))
    if log: ins.append(t(lang, "r_log"))
    ins.append(t(lang, "causal")); pts = _sample(xs.values, y.loc[xs.index].values, 1200)
    if log: eq = "ln(1 + %s) = %s + %s × %s" % (lab[target], _loc(lang, "%.3f" % a), _loc(lang, "%.5f" % b), lab[x])
    else: eq = "%s = %s + %s × %s" % (lab[target], _loc(lang, "%.2f" % a), _loc(lang, "%.4f" % b), lab[x])
    return clean(dict(model="linear", target=target, target_label=lab[target], x=x, x_label=lab[x], log_target=log, intercept=a, slope=b, slope_se=float(SE[1]), slope_p=pr, slope_ci=[float(CI[1][0]), float(CI[1][1])],
        pearson_train=r, equation=eq,
        line=[[float(x0), float(g(a + b * x0))], [float(x1), float(g(a + b * x1))]], points=pts,
        split=dict(n_train=len(tr), n_test=len(te), cutoff=str(cutoff)[:10], rule="chronological hold-out by Purchase Order Date; all items of an order on the same side"), **ev, insights=ins, robust_se="HC3"))
