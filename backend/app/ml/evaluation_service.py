"""Model evaluation & comparison: every model on the same variables, the same chronological hold-out, the same grouped cross-validation,
a monthly rolling-origin check and paired significance tests per order."""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from fastapi import HTTPException
from scipy.stats import wilcoxon, spearmanr
from sklearn.model_selection import GroupKFold
from app.core.dataset import load_schema
from app.ml.features.encoding import build
from app.ml.evaluation.metrics import metrics
from app.ml.models.base import MODEL_REGISTRY
from app.ml.models import adapters  # noqa: F401  (registers the models)
from app.ml.regression_service import _prep, select_features
from app.ml.stats.common import clean, TXT, t, _loc

TXT["en"].update(
 e_best="On the unseen test period (from {cutoff}) {model} has the lowest MAE ({mae} h) - {pct}% below the naive median baseline ({bmae} h).",
 e_none="No model beats the naive median baseline by more than 5% in MAE: with the variables known at this prediction point, lead time can hardly be predicted better than by its historical median.",
 e_cv="Grouped cross-validation on the training period {agree} the test ranking (best in cross-validation: {cvbest}).",
 e_agree="confirms", e_disagree="does not confirm",
 e_sig="The saving of {model} over the baseline is {sig}: on average {d} h per order (95% CI {lo} to {hi} h, Wilcoxon p={p}).",
 e_sigy="statistically distinguishable from zero", e_sign="not statistically distinguishable from zero",
 e_vs="{a} has on average {d} h {dir} absolute error per order than {b} (95% CI {lo} to {hi} h) - {sig}.", e_hi="higher", e_lo="lower",
 e_skew="Note: medians minimise MAE and means minimise RMSE. Because lead times are right-skewed, median-type predictions score well on MAE while mean-type regression models score better on RMSE - choose the metric that matches the business question (typical error vs avoiding very large misses).",
 e_over="{model} overfits: R²={tr} in training but {te} on the test period.",
 e_time="Monthly rolling validation: the best model by month was {list}; the MAE of {model} ranges from {mn} to {mx} h between months.",
 e_rmse="By RMSE (which penalises very long lead times more) the best model is {model}.",
 e_r2="The best R² is only {r2}: most of the variation in lead time stays unexplained by information available at this point.",
 e_pick="Recommendation for this setting: {model} (lowest test MAE).",
 e_review="⚠ Variables with unconfirmed timing are included in all models: the comparison may contain information that is not available at the prediction point.")
TXT["hu"].update(
 e_best="A nem látott teszt időszakon ({cutoff} naptól) a(z) {model} MAE-je a legalacsonyabb ({mae} h) – {pct}%-kal alacsonyabb a naiv medián alapmódszernél ({bmae} h).",
 e_none="Egyik modell sem veri meg 5%-nál nagyobb MAE-különbséggel a naiv medián alapmódszert: az ezen az előrejelzési ponton ismert változókkal az átfutási idő alig jelezhető előre jobban, mint a történeti mediánnal.",
 e_cv="A tanítóidőszakon végzett csoportos keresztvalidáció {agree} a teszt-rangsort (keresztvalidációban a legjobb: {cvbest}).",
 e_agree="megerősíti", e_disagree="nem erősíti meg",
 e_sig="A(z) {model} megtakarítása az alapmódszerhez képest {sig}: átlagosan {d} h rendelésenként (95%-os CI {lo} – {hi} h, Wilcoxon p={p}).",
 e_sigy="statisztikailag megkülönböztethető a nullától", e_sign="statisztikailag nem megkülönböztethető a nullától",
 e_vs="A(z) {a} átlagosan {d} h-val {dir} abszolút hibát ad rendelésenként, mint a(z) {b} (95%-os CI {lo} – {hi} h) – {sig}.", e_hi="nagyobb", e_lo="kisebb",
 e_skew="Megjegyzés: a medián az MAE-t, az átlag az RMSE-t minimalizálja. Mivel az átfutási idők jobbra ferdék, a medián-típusú becslések az MAE-n, az átlag-típusú regressziós modellek az RMSE-n szerepelnek jobban – azt a mérőszámot válaszd, amelyik az üzleti kérdéshez illik (tipikus hiba vs. a nagyon nagy tévedések elkerülése).",
 e_over="A(z) {model} túltanul: az R² a tanításon {tr}, a teszt időszakon {te}.",
 e_time="Havi gördülő validáció: hónaponként a legjobb modell {list} volt; a(z) {model} MAE-je a hónapok között {mn} és {mx} h között mozog.",
 e_rmse="RMSE szerint (amely a nagyon hosszú átfutásokat jobban bünteti) a legjobb modell a(z) {model}.",
 e_r2="A legjobb R² csak {r2}: az átfutási idő változékonyságának nagy részét az ezen a ponton ismert információ nem magyarázza.",
 e_pick="Javaslat erre a beállításra: {model} (legalacsonyabb teszt MAE).",
 e_review="⚠ Időzítésükben nem megerősített változók szerepelnek minden modellben: az összehasonlítás olyan információt is tartalmazhat, ami az előrejelzési ponton még nem áll rendelkezésre.")

def _cv(make, X, y, grp, k=5):
    out = []
    for a, b in GroupKFold(k).split(X, groups=grp):
        m = make().fit(X.iloc[a], y.iloc[a]); out.append(metrics(y.iloc[b], m.predict(X.iloc[b])))
    return dict(r2_mean=np.mean([o["r2"] for o in out]), r2_sd=np.std([o["r2"] for o in out]), mae_mean=np.mean([o["mae"] for o in out]), mae_sd=np.std([o["mae"] for o in out]), rmse_mean=np.mean([o["rmse"] for o in out]))

def _boot(d, n=1000, seed=0):
    rs = np.random.RandomState(seed); d = np.asarray(d); m = [d[rs.randint(0, len(d), len(d))].mean() for _ in range(n)]; return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def _wil(d):
    d = np.asarray(d); d = d[d != 0]
    try: return float(wilcoxon(d).pvalue) if len(d) > 10 else None
    except Exception: return None

def run_comparison(R, req, neg):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}; target = req.measure; lang = req.lang
    df, tr, te, cutoff = _prep(R, target, neg, req.test_fraction)
    cat, used, rejected, byid = select_features(df, tr, req, target)
    X, meta = build(df, used, tr); y = df["__y"]; ok = X.notna().all(axis=1); df, X, y = df[ok], X[ok], y[ok]; tr, te = tr.intersection(df.index), te.intersection(df.index)
    Xtr, Xte, ytr, yte = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]; grp = df.loc[tr, "d_sales_document_base"].values; region = df["d_region_code"]
    cols = list(X.columns); fgroups = {}
    for c in cols: fgroups.setdefault(meta[c]["feature"], []).append(c)
    # best single variable (chosen on the training period only)
    best_f, best_mae = None, 1e18
    for fid, cs in fgroups.items():
        sc_ = _cv(lambda cs=cs: MODEL_REGISTRY["multiple_linear"](cols=cs), Xtr, ytr, grp)["mae_mean"]
        if sc_ < best_mae: best_f, best_mae = fid, sc_
    P = dict(n_trees=150, max_depth=12, min_samples_leaf=20)
    defs = [("median_baseline", "Median baseline", "baseline", lambda: MODEL_REGISTRY["median_baseline"](), 0),
            ("region_median", "Median by region", "baseline", lambda: MODEL_REGISTRY["group_median"](region), 1),
            ("linear_single", "Linear regression (best single variable: %s)" % byid[best_f]["label"], "linear", lambda: MODEL_REGISTRY["multiple_linear"](cols=fgroups[best_f]), len(fgroups[best_f])),
            ("multiple_linear", "Multiple linear regression", "linear", lambda: MODEL_REGISTRY["multiple_linear"](), len(cols)),
            ("random_forest", "Random forest", "ensemble", lambda: MODEL_REGISTRY["random_forest"](P["n_trees"], P["max_depth"], P["min_samples_leaf"]), len(cols)),
            ("gradient_boosting", "Gradient boosting", "ensemble", lambda: MODEL_REGISTRY["gradient_boosting"](), len(cols)),
            ("gradient_boosting_median", "Gradient boosting (absolute-error loss)", "ensemble", lambda: MODEL_REGISTRY["gradient_boosting_median"](), len(cols))]
    out, preds = [], {}
    for mid, label, kind, make, nf in defs:
        t0 = time.perf_counter(); m = make().fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        pte, ptr = np.asarray(m.predict(Xte)), np.asarray(m.predict(Xtr)); preds[mid] = pte
        mt = metrics(yte, pte, nf if kind == "linear" else None); mtr = metrics(ytr, ptr)
        cv = _cv(make, Xtr, ytr, grp); mt["medae"] = float(np.median(np.abs(yte.values - pte)))
        out.append(dict(id=mid, label=label, kind=kind, n_features=nf, test=mt, train=dict(r2=mtr["r2"], mae=mtr["mae"]), cv=cv, overfit_gap_r2=(mtr["r2"] - mt["r2"]) if mtr["r2"] is not None and mt["r2"] is not None else None, fit_seconds=fit_s))
    by_mae = sorted(out, key=lambda m: m["test"]["mae"]); base = next(m for m in out if m["id"] == "median_baseline"); best = by_mae[0]
    for i, m in enumerate(by_mae): m["rank_test"] = i + 1
    for i, m in enumerate(sorted(out, key=lambda m: m["cv"]["mae_mean"])): m["rank_cv"] = i + 1
    # ---- paired per-order tests ----
    base_doc = df.loc[te, "d_sales_document_base"]; ae = {k: pd.Series(np.abs(yte.values - v), index=te).groupby(base_doc).mean() for k, v in preds.items()}
    def pair(a, b):
        d = (ae[a] - ae[b]).values; lo, hi = _boot(d); return dict(a=a, b=b, mean_saving=float(d.mean()), ci_low=lo, ci_high=hi, p=_wil(d), n_orders=int(len(d)))
    tests = [dict(model=m["id"], **pair("median_baseline", m["id"])) for m in out if m["id"] != "median_baseline"]
    others = [dict(model=m["id"], **pair(m["id"], best["id"])) for m in by_mae[1:]]
    # ---- error distribution & by region (test period) ----
    allres = np.concatenate([yte.values - v for v in preds.values()]); lo, hi = float(np.quantile(allres, .01)), float(np.quantile(allres, .99)); boxes = []
    for m in out:
        r = pd.Series(yte.values - preds[m["id"]]); q1, md, q3 = r.quantile([.25, .5, .75]); iq = q3 - q1; wl, wh = q1 - 1.5 * iq, q3 + 1.5 * iq
        boxes.append(dict(group=m["label"], id=m["id"], n=len(r), median=md, q1=q1, q3=q3, whisker_low=r[r >= wl].min(), whisker_high=r[r <= wh].max(), n_outliers=int(((r < wl) | (r > wh)).sum())))
    reg = []
    for rg, idx in df.loc[te].groupby("d_region_code").groups.items():
        pos = te.get_indexer(idx)
        if len(pos) >= 30: reg.append(dict(region=str(rg), n=int(len(pos)), mae={m["id"]: float(np.abs(yte.values[pos] - preds[m["id"]][pos]).mean()) for m in out}))
    # ---- monthly rolling origin ----
    first = df.groupby("d_sales_document_base")["Purchase Order Date"].transform("min"); month = first.dt.to_period("M").astype(str); roll = []
    for mo in sorted(month.unique())[2:]:
        tri, tei = df.index[month < mo], df.index[month == mo]
        if len(tri) < 800 or len(tei) < 100: continue
        Xf, mf = build(df, used, tri); okf = Xf.notna().all(axis=1); tri, tei = tri[okf.loc[tri].values], tei[okf.loc[tei].values]; row = dict(month=mo, n_train=int(len(tri)), n_test=int(len(tei)), mae={})
        fcols = [c for c in Xf.columns if mf[c]["feature"] == best_f]          # dummy columns differ from month to month -> use this fold's own encoding
        for mid, label, kind, make, nf in defs:
            if mid == "linear_single": mk = (lambda fc=fcols: MODEL_REGISTRY["multiple_linear"](cols=fc)) if fcols else (lambda: MODEL_REGISTRY["median_baseline"]())
            else: mk = make
            m = mk().fit(Xf.loc[tri], y.loc[tri]); row["mae"][mid] = float(np.abs(y.loc[tei].values - np.asarray(m.predict(Xf.loc[tei]))).mean())
        roll.append(row)
    # ---- generated text ----
    ins = []; pct = (1 - best["test"]["mae"] / base["test"]["mae"]) * 100
    ins.append(t(lang, "e_best", cutoff=str(cutoff)[:10], model=best["label"], mae="%.1f" % best["test"]["mae"], pct="%.0f" % pct, bmae="%.1f" % base["test"]["mae"]) if best["id"] != "median_baseline" and pct > 0 else t(lang, "e_none"))
    if best["id"] != "median_baseline" and pct <= 5: ins.append(t(lang, "e_none"))
    cvbest = min(out, key=lambda m: m["cv"]["mae_mean"]); ins.append(t(lang, "e_cv", agree=t(lang, "e_agree" if cvbest["id"] == best["id"] else "e_disagree"), cvbest=cvbest["label"]))
    bt = next((x for x in tests if x["model"] == best["id"]), None)
    if bt:
        sig = bt["p"] is not None and bt["p"] < .05 and bt["ci_low"] > 0
        ins.append(t(lang, "e_sig", model=best["label"], sig=t(lang, "e_sigy" if sig else "e_sign"), d="%.1f" % bt["mean_saving"], lo="%.1f" % bt["ci_low"], hi="%.1f" % bt["ci_high"], p=("%.1e" % bt["p"]) if bt["p"] is not None else "–"))
    if len(by_mae) > 1:
        o = others[0]; nm = {m["id"]: m["label"] for m in out}; sig = o["p"] is not None and o["p"] < .05 and (o["ci_low"] > 0 or o["ci_high"] < 0)
        ins.append(t(lang, "e_vs", a=nm[o["a"]], b=nm[o["b"]], dir=t(lang, "e_hi" if o["mean_saving"] >= 0 else "e_lo"), d="%.1f" % abs(o["mean_saving"]), lo="%.1f" % o["ci_low"], hi="%.1f" % o["ci_high"], sig=t(lang, "e_sigy" if sig else "e_sign")))
    rm = min(out, key=lambda m: m["test"]["rmse"])
    if rm["id"] != best["id"]:
        ins.append(t(lang, "e_rmse", model=rm["label"]))
        if best["kind"] == "baseline" or best["id"] == "gradient_boosting_median": ins.append(t(lang, "e_skew"))
    for m in out:
        if m["overfit_gap_r2"] is not None and m["overfit_gap_r2"] > .15: ins.append(t(lang, "e_over", model=m["label"], tr="%.2f" % m["train"]["r2"], te="%.2f" % m["test"]["r2"]))
    if roll:
        wins = [min(r["mae"], key=r["mae"].get) for r in roll]; nm = {m["id"]: m["label"] for m in out}; vals = [r["mae"][best["id"]] for r in roll]
        ins.append(t(lang, "e_time", list=", ".join("%s: %s" % (r["month"], nm[w]) for r, w in zip(roll, wins)), model=best["label"], mn="%.0f" % min(vals), mx="%.0f" % max(vals)))
    if (best["test"]["r2"] or 0) < .2: ins.append(t(lang, "e_r2", r2="%.2f" % (best["test"]["r2"] or 0)))
    ins.append(t(lang, "e_pick", model=best["label"]))
    if any(byid[u["id"]]["review"] for u in used): ins.append(t(lang, "e_review"))
    ins.append(t(lang, "causal"))
    return clean(dict(target=target, target_label=lab[target], prediction_point=req.prediction_point, models=out, best=best["id"], baseline=base["id"], best_single_variable=byid[best_f]["label"],
        split=dict(n_train=len(tr), n_test=len(te), cutoff=str(cutoff)[:10], rule="chronological hold-out by Purchase Order Date; all items of an order on the same side; grouped 5-fold CV on the training period"),
        tests_vs_baseline=tests, tests_vs_best=others, residual_boxes=boxes, residual_range=[lo, hi], by_region=reg, rolling=roll,
        used_features=[dict(id=u["id"], label=u["label"], review=u["review"]) for u in used], rejected=rejected, registry=sorted(MODEL_REGISTRY.keys()), insights=ins))
