import numpy as np, pandas as pd, statsmodels.api as sm

def fit_ols(X: pd.DataFrame, y, robust=True):
    return sm.OLS(np.asarray(y, float), sm.add_constant(X, has_constant="add")).fit(cov_type="HC3" if robust else "nonrobust")

def predict(model, X):
    return model.predict(sm.add_constant(X, has_constant="add"))

def vif(X: pd.DataFrame):
    """VIF_i = i-th diagonal of the inverse correlation matrix (exact; very large = near-perfect collinearity)."""
    if X.shape[1] < 2: return {c: 1.0 for c in X.columns}
    C = np.corrcoef(X.values, rowvar=False); C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
    d = np.diag(np.linalg.pinv(C)); return {c: float(v) for c, v in zip(X.columns, d)}
