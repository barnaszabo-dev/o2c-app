"""Lead-time prediction for one order at an explicit prediction point.
Leakage guard: only variables known at that point are used. Split: train | calibration | test in chronological order (orders never split).
Uncertainty: split-conformal style interval from signed calibration residuals; its real coverage is measured on the later test period."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from fastapi import HTTPException
from app.core.dataset import load_schema
from app.ml.features.catalog import catalog
from app.ml.features.encoding import build
from app.ml.evaluation.splits import chrono_group_split
from app.ml.evaluation.metrics import metrics
from app.ml.models.forest import ForestModel
from app.ml.models.ols import fit_ols, predict as ols_predict
from app.ml.regression_service import _prep, select_features
from app.ml.stats.common import clean, TXT, t, _loc

TXT["en"].update(
 p_pred="For an order with these characteristics the model predicts {target} of {pred} h; with {lvl}% confidence it lies between {lo} and {hi} h.",
 p_hist="Historical comparison: the median of similar past orders ({crit}; n={n}) is {smed} h (middle half: {p25}–{p75} h); the median of all orders is {amed} h.",
 p_hist_all="Historical comparison: the median of all past orders is {amed} h (middle half: {p25}–{p75} h); no sufficiently large group of similar orders was found.",
 p_48="Chance of order confirmation within 48 hours: {sshare}% among similar past orders ({share}% among all orders). The model-based estimate is {p}% (rough: it assumes the same uncertainty for every order).",
 p_48b="Model-based chance of order confirmation within 48 hours: {p}% (rough); share among all past orders: {share}%.",
 p_skew="The prediction is an average-type estimate; lead times are right-skewed, so it is higher than the typical (median) value of similar orders.",
 p_fac="Largest contributions to this prediction: {items}.",
 p_conf="Model confidence: {grade}. On unseen later orders (from {cutoff}) the model explains {r2}% of the variation (MAE {mae} h; always predicting the median: {bmae} h), and the {lvl}% interval covered {cov}% of the actual orders.",
 p_low="low", p_med="moderate", p_high="high",
 p_wide="The interval is wide ({w} h): individual lead times vary a lot, so treat the prediction as a typical range, not a promise.",
 p_cov="Warning: the interval covered only {cov}% of later orders (nominal {lvl}%); lead-time behaviour changed over time, so the real uncertainty is larger than shown.",
 p_unseen="Warning: {items} was not seen often enough in the training data and is treated as 'Other'.",
 p_range="Warning: {items} lies outside the range of the training data (1st–99th percentile); the prediction is an extrapolation.",
 p_trained="The model was trained on orders created until {tr_end}, calibrated on the following period and tested on orders from {cutoff}. Refit it regularly with new data.",
 p_whatif="Changing an input shows how the estimate differs between types of orders; it does not show what would happen if a specific order were changed.")
TXT["hu"].update(
 p_pred="Az ilyen jellemzőjű rendelésre a modell {target} idejét {pred} h-ra becsüli; {lvl}%-os bizonyossággal {lo} és {hi} h között lesz.",
 p_hist="Történeti összevetés: a hasonló korábbi rendelések mediánja ({crit}; n={n}) {smed} h (középső fele: {p25}–{p75} h); az összes rendelés mediánja {amed} h.",
 p_hist_all="Történeti összevetés: az összes korábbi rendelés mediánja {amed} h (középső fele: {p25}–{p75} h); elég nagy hasonló csoportot nem találtunk.",
 p_48="A rendelés 48 órán belüli visszaigazolásának esélye: a hasonló korábbi rendelések {sshare}%-a, az összes rendelés {share}%-a. A modell alapú becslés {p}% (durva: minden rendelésre azonos bizonytalanságot feltételez).",
 p_48b="A rendelés 48 órán belüli visszaigazolásának modell alapú esélye: {p}% (durva); az összes korábbi rendelés aránya: {share}%.",
 p_skew="Az előrejelzés átlag-típusú becslés; az átfutási idők jobbra ferdék, ezért magasabb a hasonló rendelések tipikus (medián) értékénél.",
 p_fac="Az előrejelzéshez a legnagyobb hozzájárulást adják: {items}.",
 p_conf="A modell megbízhatósága: {grade}. A később leadott, nem látott rendeléseken ({cutoff} naptól) a változékonyság {r2}%-át magyarázza (MAE {mae} h; a mediánt becsülve: {bmae} h), és a {lvl}%-os sáv a tényleges rendelések {cov}%-át fedte le.",
 p_low="alacsony", p_med="közepes", p_high="magas",
 p_wide="A sáv széles ({w} h): az egyes átfutási idők nagyon különböznek, ezért az előrejelzést tipikus tartományként értelmezd, ne ígéretként.",
 p_cov="Figyelem: a sáv a későbbi rendeléseknek csak {cov}%-át fedte le (névleges {lvl}%); az átfutási idők viselkedése az idő során változott, ezért a valódi bizonytalanság nagyobb a mutatottnál.",
 p_unseen="Figyelem: a(z) {items} a tanítóadatban nem fordult elő elég gyakran, ezért „Egyéb”-ként kezeljük.",
 p_range="Figyelem: a(z) {items} kívül esik a tanítóadat tartományán (1.–99. percentilis); az előrejelzés extrapoláció.",
 p_trained="A modellt a {tr_end} napig létrehozott rendeléseken tanítottuk, a következő időszakon kalibráltuk, és a {cutoff} naptól érkezett rendeléseken teszteltük. Új adatokkal rendszeresen tanítsd újra.",
 p_whatif="Egy bemenet módosítása azt mutatja, hogyan tér el a becslés a különböző típusú rendelések között; nem azt, hogy mi történne, ha egy konkrét rendelést megváltoztatnánk.")

def _raw_levels(s):
    return s.astype("string").fillna("Missing")

def run_prediction(R, req, neg):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}; ev = sc["events"]; target = req.measure; lang = req.lang
    tm = next(m for m in sc["measures"] if m["id"] == target); pe = ev.index(sc["prediction_points"][req.prediction_point]["event"])
    if not tm["end_event"] or ev.index(tm["end_event"]) <= pe:
        valid = [lab[m["id"]] for m in sc["measures"] if m["end_event"] and ev.index(m["end_event"]) > pe]
        raise HTTPException(422, "%s is not a valid target at this prediction point (it must end after it). Valid targets: %s" % (lab[target], ", ".join(valid)))
    df, trall, te, cutoff = _prep(R, target, neg, 0.2)
    cat, used, rejected, byid = select_features(df, trall, req, target)
    tr, cal, _ = chrono_group_split(df.loc[trall], 0.25)
    if len(cal) < 100: raise HTTPException(422, "Too few observations for a calibration period")
    cols = [u["column"] for u in used]; lvl = min(max(req.level, 0.5), 0.99)
    # ---- form definition: allowed variables, levels, defaults (from the training period) ----
    fields, default = [], {}
    for u in used:
        s = df.loc[tr, u["column"]]
        if u["type"] == "categorical":
            vc = _raw_levels(s).value_counts(); default[u["id"]] = str(vc.idxmax())
            fields.append(dict(id=u["id"], label=u["label"], type="categorical", review=u["review"], default=default[u["id"]], levels=[dict(level=str(k), n=int(v)) for k, v in vc.head(25).items()]))
        else:
            x = pd.to_numeric(s, errors="coerce").dropna(); default[u["id"]] = float(x.median())
            fields.append(dict(id=u["id"], label=u["label"], type="numeric", review=u["review"], default=default[u["id"]], p01=float(x.quantile(.01)), p99=float(x.quantile(.99))))
    # ---- inputs (missing -> default), warnings ----
    inp, warn_unseen, warn_range = {}, [], []
    for f in fields:
        v = (req.inputs or {}).get(f["id"])
        if f["type"] == "categorical":
            inp[f["id"]] = str(v) if v not in (None, "") else f["default"]
            if inp[f["id"]] not in [l["level"] for l in f["levels"]] or next((l["n"] for l in f["levels"] if l["level"] == inp[f["id"]]), 0) < 30: warn_unseen.append("%s = %s" % (f["label"], inp[f["id"]]))
        else:
            try: inp[f["id"]] = float(v)
            except (TypeError, ValueError): inp[f["id"]] = f["default"]
            if inp[f["id"]] < f["p01"] or inp[f["id"]] > f["p99"]: warn_range.append("%s = %s" % (f["label"], inp[f["id"]]))
    # ---- design matrix: data rows + the new order + one "typical value" variant per variable (for the contributions) ----
    variants = [dict(inp)] + [dict(inp, **{f["id"]: f["default"]}) for f in fields]
    new = pd.DataFrame([{u["column"]: (np.nan if (u["type"] == "categorical" and r[u["id"]] == "Missing") else r[u["id"]]) for u in used} for r in variants])
    new.index = np.arange(df.index.max() + 1, df.index.max() + 1 + len(new)); new = new.astype(object)
    ext = pd.concat([df[cols].astype(object), new]); X, meta = build(ext, used, tr)
    ok = X.loc[df.index].notna().all(axis=1); tr, cal, te = tr[ok.loc[tr].values], cal[ok.loc[cal].values], te[ok.loc[te].values]
    y = df["__y"]; Xn = X.loc[new.index]
    if req.model == "ols":
        m = fit_ols(X.loc[tr], y.loc[tr], robust=False); pf = lambda Z: np.asarray(ols_predict(m, Z)); rf = None
    else:
        rf = ForestModel(150, 12, 20).fit(X.loc[tr].values, y.loc[tr].values); pf = lambda Z: rf.predict(np.asarray(Z))
    pcal, pte = pf(X.loc[cal]), pf(X.loc[te]); res = y.loc[cal].values - pcal; a = (1 - lvl) / 2; qlo, qhi = np.quantile(res, [a, 1 - a])
    cover = float(np.mean((y.loc[te].values >= pte + qlo) & (y.loc[te].values <= pte + qhi))); mte = metrics(y.loc[te], pte); mb = metrics(y.loc[te], np.full(len(te), float(y.loc[tr].median())))
    p0 = float(pf(Xn.iloc[[0]])[0]); lo, hi = max(0.0, p0 + qlo), p0 + qhi
    p48 = None
    if target in ("po_to_oc", "so_to_oc", "po_to_oc_agg"): p48 = float(np.mean(p0 + res <= 48)); share48 = float((y <= 48).mean())
    # ---- contributions: SHAP (random forest) or "variable set to its typical value" (occlusion) ----
    method, contrib = "occlusion", []
    groups = {}
    for i, c in enumerate(X.columns): groups.setdefault(meta[c]["feature"], []).append(i)
    pv = pf(Xn.values if hasattr(Xn, "values") else Xn)
    if rf is not None:
        try:
            import shap
            sv = np.asarray(shap.TreeExplainer(rf.m).shap_values(Xn.iloc[[0]].values, check_additivity=False))[0]; method = "shap"
            for f in fields: contrib.append(dict(id=f["id"], label=f["label"], value=inp[f["id"]], contribution=float(sv[groups[f["id"]]].sum())))
        except Exception: contrib = []
    if not contrib:
        for k, f in enumerate(fields): contrib.append(dict(id=f["id"], label=f["label"], value=inp[f["id"]], contribution=float(pv[0] - pv[k + 1])))
    contrib.sort(key=lambda c: -abs(c["contribution"]))
    # ---- historical comparison (all past orders of the selection) ----
    hy = df["__y"]; allm = dict(n=int(len(hy)), median=float(hy.median()), p25=float(hy.quantile(.25)), p75=float(hy.quantile(.75)))
    crit = [f for f in fields if f["type"] == "categorical" and inp[f["id"]] != "Missing"]; sim = None
    while crit:
        mask = pd.Series(True, index=df.index)
        for f in crit: mask &= _raw_levels(df[next(u["column"] for u in used if u["id"] == f["id"])]) == inp[f["id"]]
        if mask.sum() >= 30:
            sy = hy[mask]; sim = dict(n=int(len(sy)), median=float(sy.median()), p25=float(sy.quantile(.25)), p75=float(sy.quantile(.75)), share_48=float((sy <= 48).mean()), criteria=["%s = %s" % (f["label"], inp[f["id"]]) for f in crit]); break
        crit = crit[:-1]
    # ---- confidence + generated text ----
    r2 = mte["r2"] or 0; grade = "p_low" if r2 < .2 else ("p_med" if r2 < .5 else "p_high"); width = hi - lo
    ins = [t(lang, "p_pred", target=lab[target], pred="%.0f" % p0, lvl="%.0f" % (lvl * 100), lo="%.0f" % lo, hi="%.0f" % hi)]
    ins.append(t(lang, "p_hist", crit=", ".join(sim["criteria"]), n=sim["n"], smed="%.0f" % sim["median"], p25="%.0f" % sim["p25"], p75="%.0f" % sim["p75"], amed="%.0f" % allm["median"]) if sim
               else t(lang, "p_hist_all", amed="%.0f" % allm["median"], p25="%.0f" % allm["p25"], p75="%.0f" % allm["p75"]))
    if p48 is not None: ins.append(t(lang, "p_48", p="%.0f" % (p48 * 100), share="%.0f" % (share48 * 100), sshare="%.0f" % (sim["share_48"] * 100)) if sim else t(lang, "p_48b", p="%.0f" % (p48 * 100), share="%.0f" % (share48 * 100)))
    if sim and p0 > 1.15 * sim["median"]: ins.append(t(lang, "p_skew"))
    ins.append(t(lang, "p_fac", items="; ".join("%s = %s (%s%s h)" % (c["label"], c["value"] if isinstance(c["value"], str) else "%.1f" % c["value"], "+" if c["contribution"] > 0 else "", _loc(lang, "%.1f" % c["contribution"])) for c in contrib[:3])))
    ins.append(t(lang, "p_conf", grade=t(lang, grade), cutoff=str(cutoff)[:10], r2="%.0f" % (max(r2, 0) * 100), mae="%.0f" % mte["mae"], bmae="%.0f" % mb["mae"], lvl="%.0f" % (lvl * 100), cov="%.0f" % (cover * 100)))
    if cover < lvl - .1: ins.append(t(lang, "p_cov", cov="%.0f" % (cover * 100), lvl="%.0f" % (lvl * 100)))
    if width > 1.5 * max(allm["median"], 1): ins.append(t(lang, "p_wide", w="%.0f" % width))
    if warn_unseen: ins.append(t(lang, "p_unseen", items=", ".join(warn_unseen)))
    if warn_range: ins.append(t(lang, "p_range", items=", ".join(warn_range)))
    if any(f["review"] for f in fields): ins.append(t(lang, "f_review"))
    ins += [t(lang, "p_trained", tr_end=str(df.loc[tr, "Purchase Order Date"].max())[:10], cutoff=str(cutoff)[:10]), t(lang, "p_whatif"), t(lang, "causal")]
    return clean(dict(model=req.model, target=target, target_label=lab[target], prediction_point=req.prediction_point, level=lvl,
        prediction=dict(value=p0, lower=lo, upper=hi, p_within_48h=p48, share_within_48h=share48 if p48 is not None else None),
        history=dict(all=allm, similar=sim), contributions=contrib, contribution_method=method,
        confidence=dict(grade=grade[2:], test=mte, baseline_mae=mb["mae"], coverage_test=cover, coverage_nominal=lvl, interval_width=width, residual_quantiles=[float(qlo), float(qhi)]),
        split=dict(n_train=len(tr), n_calibration=len(cal), n_test=len(te), cutoff=str(cutoff)[:10]),
        form=dict(fields=fields, values=inp), rejected=rejected, warnings=dict(unseen=warn_unseen, out_of_range=warn_range), insights=ins))
