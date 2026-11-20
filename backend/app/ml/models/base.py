"""Extension point: every future algorithm (linear, multiple linear, random forest, gradient boosting, ...) implements this interface
and registers itself, so the API/evaluation layers never import concrete models."""
from abc import ABC, abstractmethod
MODEL_REGISTRY: dict = {}

def register(name):
    def deco(cls): MODEL_REGISTRY[name] = cls; return cls
    return deco

class BaseLeadTimeModel(ABC):
    name = "base"
    @abstractmethod
    def fit(self, X, y): ...
    @abstractmethod
    def predict(self, X): ...
    def predict_interval(self, X, alpha=0.1): raise NotImplementedError
    def feature_importance(self): raise NotImplementedError
    def get_params(self) -> dict: return {}
