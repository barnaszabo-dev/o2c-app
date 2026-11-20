"""Random forest regressor behind the common model interface (registered as 'random_forest')."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from app.ml.models.base import BaseLeadTimeModel, register

@register("random_forest")
class ForestModel(BaseLeadTimeModel):
    name = "random_forest"
    def __init__(self, n_trees=150, max_depth=12, min_samples_leaf=20, seed=42):
        self.params = dict(n_estimators=n_trees, max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_features=0.5, n_jobs=-1, random_state=seed)
        self.m = RandomForestRegressor(**self.params)
    def fit(self, X, y): self.m.fit(X, y); return self
    def predict(self, X): return self.m.predict(X)
    def feature_importance(self): return self.m.feature_importances_
    def get_params(self): return dict(self.params)
