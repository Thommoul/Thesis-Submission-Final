"""GPU detection for XGBoost — uses CUDA when available, otherwise falls back to CPU."""

import functools

import numpy as np


@functools.lru_cache(maxsize=1)
def get_xgb_device():
    from xgboost import XGBRegressor

    try:
        model = XGBRegressor(n_estimators=1, max_depth=1, device="cuda", tree_method="hist")
        model.fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
        return "cuda"
    except Exception:
        return "cpu"
