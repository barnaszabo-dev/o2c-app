import numpy as np

def metrics(y, p, n_features=None):
    y, p = np.asarray(y, float), np.asarray(p, float); n = len(y); res = y - p
    sst = ((y - y.mean()) ** 2).sum(); r2 = 1 - (res ** 2).sum() / sst if sst else None
    adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1) if (r2 is not None and n_features is not None and n > n_features + 1) else None
    return dict(n=n, r2=r2, adj_r2=adj, mae=np.abs(res).mean(), rmse=np.sqrt((res ** 2).mean()))
