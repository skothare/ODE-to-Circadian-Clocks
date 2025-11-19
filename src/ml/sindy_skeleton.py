"""
PySINDy skeleton: load processed time series and attempt to discover sparse ODE structure.
"""
try:
    import pysindy as ps
except Exception:
    ps = None

import numpy as np


def find_sindy_model(X, t=None):
    if ps is None:
        raise RuntimeError("pysindy not installed; add to requirements to run this function")
    model = ps.SINDy()
    model.fit(X, t=t)
    return model
