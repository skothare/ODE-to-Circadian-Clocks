"""
PySINDy skeleton: load processed time series and attempt to discover sparse ODE structure.
"""
"""try:
    import pysindy as ps
except Exception:
    ps = None

import numpy as np


def find_sindy_model(X, t=None):
    if ps is None:
        raise RuntimeError("pysindy not installed; add to requirements to run this function")
    model = ps.SINDy()
    model.fit(X, t=t)
    return model"""


"""
SINDy skeleton with Leloup–Goldbeter–like nonlinearities:
- Hill activation
- Hill inhibition
- Michaelis–Menten saturation
This library is compatible with multi-dimensional inputs.
"""

import numpy as np
try:
    import pysindy as ps
except Exception:
    ps = None

from scipy.signal import savgol_filter


# ============================================================
#   LG-style nonlinearities
# ============================================================

def hill_act_vec(x, K=1.0, n=2.0):
    return (x**n) / (K**n + x**n + 1e-12)

def hill_inh_vec(x, K=1.0, n=4.0):
    return 1.0 / (1.0 + (x / (K + 1e-12)) ** n)

def mm_vec(x, V=1.0, K=0.1):
    return V * x / (K + x + 1e-12)


# ============================================================
#   Custom single-variable nonlinear library
#   (No interactions here — handled separately)
# ============================================================

class LGLibrary(ps.CustomLibrary):
    def __init__(self):
        super().__init__(
            library_functions=[
                lambda x: x,
                lambda x: x**2,
                hill_act_vec,
                hill_inh_vec,
                mm_vec
            ],
            function_names=[
                "x",
                "x^2",
                "hill_act",
                "hill_inh",
                "mm"
            ]
        )


# ============================================================
#   Build SINDy model
# ============================================================

def build_sindy_model(threshold=0.01, smooth=True):
    if ps is None:
        raise RuntimeError("PySINDy not installed. Run: pip install pysindy")

    # Linear + interaction terms (matching LG cross-terms)
    library_linear = ps.PolynomialLibrary(degree=1, include_interaction=True)

    # LG nonlinearities (univariate)
    library_lg = LGLibrary()

    # Combine
    full_library = ps.GeneralizedLibrary([library_linear, library_lg])

    differentiation = ps.SmoothedFiniteDifference(
        smoother_kws={"window_length": 11, "polyorder": 3, "axis": 0}
    )

    optimizer = ps.STLSQ(threshold=threshold)

    model = ps.SINDy(
        feature_library=full_library,
        differentiation_method=differentiation,
        optimizer=optimizer
    )

    return model


# ============================================================
#   SINDy fitting wrapper
# ============================================================

def sindy_fit(X, t, threshold=0.01, smooth=True):
    if smooth:
        X = savgol_filter(X, window_length=11, polyorder=3, axis=0)

    model = build_sindy_model(threshold=threshold, smooth=smooth)
    model.fit(X, t=t)
    return model
