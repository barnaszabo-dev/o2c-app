"""Random forest regression: same split, same leakage guard, same encoding and same metrics as the linear models, plus a side-by-side comparison."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from fastapi import HTTPException
from sklearn.model_selection import GroupKFold
from app.core.dataset import load_schema
from app.ml.features.encoding import build
from app.ml.evaluation.metrics import metrics
from app.ml.models.forest import ForestModel
from app.ml.regression_service import _prep, select_features, _fit_eval, _hist, _sample
from app.ml.stats.common import clean, TXT, t, _loc

TXT["en"].update(
 f_fit="The random forest ({n} trees, depth ≤ {depth}, ≥ {leaf} orders per leaf) reaches R²={r2}, MAE={mae} h and RMSE={rmse} h on unseen later orders (from {cutoff}); the naive training median gives MAE={bmae} h.",
 f_cmp="Against the linear regression on the same variables and split, the forest {verdict} (MAE {fmae} h vs {lmae} h; R² {fr2} vs {lr2}).",
 f_better="is better by {pct}% in MAE", f_worse="is worse by {pct}% in MAE", f_same="performs about the same",
 f_imp="Most influential variables for the forest's predictions (permutation importance on the test period, increase in MAE when the variable is shuffled): {items}.",
 f_over="Overfitting warning: R² is {tr} in training but {te} on the test period (gap {gap}). Increase the minimum orders per leaf or reduce the depth.",
 f_drift="Cross-validation inside the training period gives R²={cv}, the later test period {te}: the relationships are not fully stable over time.",
 f_imp_note="Importance describes what the model uses for prediction, not what causes long lead times.",
 f_review="⚠ Variables with unconfirmed timing are included: the result may contain information that is not available at the prediction point.")
TXT["hu"].update(
 f_fit="A random forest ({n} fa, mélység ≤ {depth}, ≥ {leaf} rendelés levelenként) a később leadott, nem látott rendeléseken ({cutoff} naptól) R²={r2}, MAE={mae} h és RMSE={rmse} h értéket ér el; a naiv tanító-medián MAE-je {bmae} h.",
 f_cmp="Ugyanazokon a változókon és ugyanazon a bontáson a lineáris regresszióhoz képest az erdő {verdict} (MAE {fmae} h vs {lmae} h; R² {fr2} vs {lr2}).",
 f_better="{pct}%-kal jobb az MAE-ben", f_worse="{pct}%-kal rosszabb az MAE-ben", f_same="nagyjából ugyanannyit teljesít",
 f_imp="Az erdő előrejelzéseit leginkább befolyásoló változók (permutációs fontosság a teszt időszakon, az MAE növekedése, ha a változót összekeverjük): {items}.",
 f_over="Túltanulás figyelmeztetés: az R² a tanításon {tr}, a teszt időszakon {te} (különbség {gap}). Növeld a levelenkénti minimális rendelésszámot vagy csökkentsd a mélységet.",
 f_drift="A tanítóidőszakon belüli keresztvalidáció R²={cv}, a későbbi teszt időszak {te}: az összefüggések az idő során nem teljesen stabilak.",
 f_imp_note="A fontosság azt mutatja, mit használ a modell az előrejelzéshez, nem azt, hogy mi okozza a hosszú átfutást.",
 f_review="⚠ Időzítésükben nem megerősített változók is szerepelnek: az eredmény olyan információt is tartalmazhat, ami az előrejelzési ponton még nem áll rendelkezésre.")

def _group_cols(meta, cols):
    g = {}
    for i, c in enumerate(cols): g.setdefault(meta[c]["feature"], []).append(i)
    return g

def run_forest(R, req, neg):
    sc = load_schema(); lab = {m["id"]: m["label"] for m in sc["measures"]}; target = req.measure; lang = req.lang
    nt, md, ml = max(10, min(req.n_trees, 500)), (None if req.max_depth in (None, 0) else max(2, min(req.max_depth, 40))), max(1, min(req.min_samples_leaf, 500))
    df, tr, te, cutoff = _prep(R, target, neg, req.test_fraction)
    cat, used, rejected, byid = select_features(df, tr, req, target)
    X, meta = build(df, used, tr); y = df["__y"]
    ok = X.notna().all(axis=1); dropped = int((~ok).sum()); df, X, y = df[ok], X[ok], y[ok]; tr, te = tr.intersection(df.index), te.intersection(df.index)
    log = req.log_target; g = (lambda v: np.log1p(v)) if log else (lambda v: v); ginv = (lambda v: np.expm1(v)) if log else (lambda v: v)
    Xtr, Xte, ytr, yte = X.loc[tr], X.loc[te], y.loc[tr], y.loc[te]
    rf = ForestModel(nt, md, ml).fit(Xtr.values, g(ytr).values); pte = ginv(rf.predict(Xte.values)); ptr = ginv(rf.predict(Xtr.values))
    mte, mtr, mb = metrics(yte, pte), metrics(ytr, ptr), metrics(yte, np.full(len(yte), float(ytr.median())))
    cv, grp = [], df.loc[tr, "d_sales_document_base"].values
    for a, b in GroupKFold(5).split(Xtr, groups=grp):
        f = ForestModel(nt, md, ml).fit(Xtr.iloc[a].values, g(ytr.iloc[a]).values); cv.append(metrics(ytr.iloc[b], ginv(f.predict(Xtr.iloc[b].values))))
    cvs = dict(folds=cv, r2_mean=np.mean([c["r2"] for c in cv]), r2_sd=np.std([c["r2"] for c in cv]), mae_mean=np.mean([c["mae"] for c in cv]), rmse_mean=np.mean([c["rmse"] for c in cv]))
    # ---- importance: impurity (per variable) and permutation on the TEST period (variable's dummies shuffled together) ----
    groups = _group_cols(meta, list(X.columns)); imp = rf.feature_importance(); rs = np.random.RandomState(0); base_mae = mte["mae"]; perm = []
    for fid, idx in groups.items():
        incs = []
        for _ in range(3):
            Xp = Xte.values.copy(); Xp[:, idx] = Xp[rs.permutation(len(Xp))][:, idx]; incs.append(np.abs(yte.values - ginv(rf.predict(Xp))).mean() - base_mae)
        perm.append(dict(id=fid, label=byid[fid]["label"], mae_increase=np.mean(incs), sd=np.std(incs), impurity=float(imp[idx].sum())))
    perm.sort(key=lambda d: -d["mae_increase"])
    # ---- linear regression on the same variables and split (fair comparison) ----
    _, lin, _ = _fit_eval(df, tr, te, X, y, log, lang)
    res = np.asarray(yte) - pte
    comparison = [dict(model="baseline_median", r2=mb["r2"], mae=mb["mae"], rmse=mb["rmse"], cv_r2=None, cv_mae=None),
                  dict(model="multiple_linear", r2=lin["test"]["r2"], mae=lin["test"]["mae"], rmse=lin["test"]["rmse"], cv_r2=lin["cv"]["r2_mean"], cv_mae=lin["cv"]["mae_mean"]),
                  dict(model="random_forest", r2=mte["r2"], mae=mte["mae"], rmse=mte["rmse"], cv_r2=cvs["r2_mean"], cv_mae=cvs["mae_mean"])]
    d_mae = (lin["test"]["mae"] - mte["mae"]) / lin["test"]["mae"] * 100
    verdict = t(lang, "f_same") if abs(d_mae) < 1 else (t(lang, "f_better", pct=f"{d_mae:.0f}") if d_mae > 0 else t(lang, "f_worse", pct=f"{-d_mae:.0f}"))
    ins = [t(lang, "f_fit", n=nt, depth=md or "∞", leaf=ml, r2=f"{mte['r2']:.2f}", mae=f"{mte['mae']:.1f}", rmse=f"{mte['rmse']:.1f}", cutoff=str(cutoff)[:10], bmae=f"{mb['mae']:.1f}"),
           t(lang, "f_cmp", verdict=verdict, fmae=f"{mte['mae']:.1f}", lmae=f"{lin['test']['mae']:.1f}", fr2=f"{mte['r2']:.2f}", lr2=f"{lin['test']['r2']:.2f}")]
    top = [p for p in perm if p["mae_increase"] > 0][:3]
    def _item(p): return "%s (+%s h)" % (p["label"], _loc(lang, "%.1f" % p["mae_increase"]))
    if top: ins.append(t(lang, "f_imp", items="; ".join(_item(p) for p in top)))
    gap = mtr["r2"] - mte["r2"]
    if gap > .15: ins.append(t(lang, "f_over", tr=f"{mtr['r2']:.2f}", te=f"{mte['r2']:.2f}", gap=f"{gap:.2f}"))
    if cvs["r2_mean"] - mte["r2"] > .08: ins.append(t(lang, "f_drift", cv=f"{cvs['r2_mean']:.2f}", te=f"{mte['r2']:.2f}"))
    if any(byid[u["id"]]["review"] for u in used): ins.append(t(lang, "f_review"))
    ins += [t(lang, "f_imp_note"), t(lang, "causal")]
    return clean(dict(model="random_forest", target=target, target_label=lab[target], prediction_point=req.prediction_point, log_target=log, n_features=X.shape[1],
        params=dict(n_trees=nt, max_depth=md, min_samples_leaf=ml, max_features=0.5), overfit_gap_r2=gap,
        split=dict(n_train=len(tr), n_test=len(te), cutoff=str(cutoff)[:10], rule="chronological hold-out by Purchase Order Date; all items of an order on the same side", dropped_rows_missing_numeric=dropped),
        used_features=[dict(id=u["id"], label=u["label"], type=u["type"], review=u["review"]) for u in used], rejected=rejected,
        test=mte, train=mtr, baseline_median=dict(value=float(ytr.median()), **mb), cv=cvs, importance=perm, comparison=comparison,
        residual_hist=_hist(res), actual_vs_pred=_sample(np.asarray(yte), pte), resid_vs_fitted=_sample(pte, res), insights=ins))
