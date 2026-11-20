"""Model adapters behind the common interface (registered by name). New algorithms only need to be registered here; the evaluation page picks them up."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from app.ml.models.base import BaseLeadTimeModel, register
from app.ml.models import forest        # registers 'random_forest'
from app.ml.models.ols import fit_ols, predict as ols_predict

@register("median_baseline")
class MedianModel(BaseLeadTimeModel):
    name = "median_baseline"
    def fit(self, X, y): self.v = float(np.median(y)); return self
    def predict(self, X): return np.full(len(X), self.v)

@register("group_median")
class GroupMedianModel(BaseLeadTimeModel):
    """Median of the training period per category (default: region); unknown category -> global median."""
    name = "group_median"
    def __init__(self, groups: pd.Series): self.g = groups
    def fit(self, X, y):
        d = pd.DataFrame({"g": self.g.reindex(X.index).values, "y": np.asarray(y)}); self.m = d.groupby("g")["y"].median().to_dict(); self.v = float(np.median(y)); return self
    def predict(self, X): return np.array([self.m.get(k, self.v) for k in self.g.reindex(X.index).values])

@register("multiple_linear")
class OLSModel(BaseLeadTimeModel):
    name = "multiple_linear"
    def __init__(self, cols=None): self.cols = cols
    def _x(self, X): return X[self.cols] if self.cols else X
    def fit(self, X, y): self.m = fit_ols(self._x(X), y, robust=False); return self
    def predict(self, X): return np.asarray(ols_predict(self.m, self._x(X)))

@register("gradient_boosting")
class BoostingModel(BaseLeadTimeModel):
    name = "gradient_boosting"
    def __init__(self): self.m = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0)
    def fit(self, X, y): self.m.fit(np.asarray(X), np.asarray(y)); return self
    def predict(self, X): return self.m.predict(np.asarray(X))
    def get_params(self): return self.m.get_params()

@register("gradient_boosting_median")
class BoostingMedianModel(BoostingModel):
    """Same booster trained on the absolute-error loss = a conditional-median model (fair opponent for a median baseline under MAE)."""
    name = "gradient_boosting_median"
    def __init__(self): self.m = HistGradientBoostingRegressor(loss="absolute_error", max_depth=6, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0)
