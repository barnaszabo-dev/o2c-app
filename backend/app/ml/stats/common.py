import math, re
import numpy as np, pandas as pd

def clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if (math.isnan(o) or math.isinf(o)) else float(f"{o:.6g}")   # significant digits: keeps tiny p-values (2.5e-12) intact
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o

def series(R: pd.DataFrame, mid: str, exclude_negative: bool) -> pd.Series:
    s = pd.to_numeric(R[f"h_{mid}"], errors="coerce").dropna()
    return s[s >= 0] if exclude_negative else s

def describe(s: pd.Series) -> dict:
    if s.empty: return dict(n=0)
    q1, q3 = s.quantile(.25), s.quantile(.75); iqr = q3 - q1; lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return dict(n=len(s), median=s.median(), mean=s.mean(), sd=s.std() if len(s) > 1 else None, min=s.min(), p25=q1, p75=q3, p95=s.quantile(.95), max=s.max(),
                skew=s.skew() if len(s) > 2 else None, n_negative=int((s < 0).sum()), n_outlier_high=int((s > hi).sum()), n_outlier_low=int((s < lo).sum()),
                whisker_low=s[s >= lo].min(), whisker_high=s[s <= hi].max())

TXT = {  # localized, templated insights (generated from computed numbers only)
 "en": dict(
  size="{n} order items ({d} distinct orders) are in the current selection.",
  skew="For {m} the mean ({mean} h) is {ratio}x the median ({med} h) and {pct}% of values are high outliers (Tukey rule): the distribution is right-skewed, so the median is the more representative indicator.",
  grp="By {dim}: highest median {hi} ({hv} h), lowest {lo} ({lv} h), a difference of {diff} h (groups with at least 30 orders).",
  pair="Strongest relationship between distinct non-overlapping variables: {a} and {b} (Spearman {rho}, Pearson {r}, n={n}).",
  struct="Note: {a} contains {b} as a component, so a high correlation between them is structural, not an insight.",
  assoc="Strongest association with {m}: {v} (eta={eta}, explains about {pct}% of variance; Kruskal-Wallis p={p}).",
  weak="Variables with eta below 0.1 explain less than 1% of the variance on their own.",
  bign="With large samples almost every p-value is tiny; judge relevance by effect size (eta, correlation), not by significance alone.",
  causal="These are statistical associations, not causal effects. Predictive importance (later modules) and causation are different things."),
 "hu": dict(
  size="A jelenlegi kijelölésben {n} rendelési tétel ({d} különböző rendelés) szerepel.",
  skew="{m} esetén az átlag ({mean} h) a medián ({med} h) {ratio}-szorosa, és az értékek {pct}%-a felső kiugró érték (Tukey-szabály): az eloszlás jobbra ferde, ezért a medián a reprezentatívabb mutató.",
  grp="{dim} szerint: a legmagasabb medián {hi} ({hv} h), a legalacsonyabb {lo} ({lv} h), a különbség {diff} h (legalább 30 rendelést tartalmazó csoportok).",
  pair="A legerősebb kapcsolat egymást nem átfedő változók között: {a} és {b} (Spearman {rho}, Pearson {r}, n={n}).",
  struct="Megjegyzés: a(z) {a} tartalmazza a(z) {b} lépést, ezért a köztük lévő magas korreláció szerkezeti, nem felismerés.",
  assoc="A legerősebb összefüggés a(z) {m} mutatóval: {v} (eta={eta}, a variancia kb. {pct}%-át magyarázza; Kruskal–Wallis p={p}).",
  weak="A 0,1 alatti eta értékű változók önmagukban a variancia 1%-át sem magyarázzák.",
  bign="Nagy mintánál szinte minden p-érték elenyésző; a relevanciát a hatásméret (eta, korreláció) mutatja, nem a szignifikancia.",
  causal="Ezek statisztikai összefüggések, nem ok-okozati hatások. A prediktív fontosság (későbbi modulok) és az ok-okozatiság különböző dolog."),
}
DIM_LABEL = {"en": dict(region="Region", allocation_group="Allocation group", customer_category="T1/T2/RS", early="Early / Non-early", company="Company"),
             "hu": dict(region="Régió", allocation_group="Allokációs csoport", customer_category="T1/T2/RS", early="Korai / nem korai", company="Vállalat")}
def pfmt(p): return "<1e-300" if p is not None and p < 1e-300 else f"{p:.1e}"
TXT["en"]["early_note"] = "Early / Non-early is a structural split (early orders are placed well ahead of their allocation). When its timing relative to the order is confirmed it can be used as a feature; until then treat this association with care."
TXT["hu"]["early_note"] = "A korai / nem korai egy szerkezeti felosztás (a korai rendelések jóval az allokációjuk előtt kerülnek leadásra). Ha az időzítése a rendeléshez képest megerősítést nyer, feature-ként használható; addig az összefüggést óvatosan kell kezelni."
_NUM = re.compile(r"^[<\-]?[\d,.]+(e[+-]?\d+)?$")
def _loc(lang, v):
    """Hungarian number format: 14 601 and 0,35 (only for pure numeric strings)."""
    v = str(v)
    return v.replace(",", "\u00a0").replace(".", ",") if lang == "hu" and _NUM.match(v) else v
def t(lang, key, **kw):
    return TXT.get(lang, TXT["en"])[key].format(**{k: _loc(lang, v) for k, v in kw.items()})
