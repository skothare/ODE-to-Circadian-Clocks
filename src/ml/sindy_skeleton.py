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
Designed to be numerically safe on PCA coordinates (which can be negative).
"""

import numpy as np
try:
    import pysindy as ps
except Exception:
    ps = None

from scipy.signal import savgol_filter


# ============================================================
#   LG-style nonlinearities (NaN-safe)
# ============================================================

def hill_act_vec(x, K=1.0, n=2):
    """Hill activation: x^n / (K^n + x^n)"""
    x_clamped = np.maximum(x, 0)      # <-- actual clamping
    x_n = np.power(x_clamped, n)
    return x_n / (np.power(K, n) + x_n + 1e-12)


def hill_inh_vec(x, K=1.0, n=4):
    """Hill inhibition: 1 / (1 + (x/K)^n)"""
    x_clamped = np.maximum(x, 0)
    frac = x_clamped / (K + 1e-12)
    x_n = np.power(frac, n)
    return 1.0 / (1.0 + x_n)


def mm_vec(x, V=1.0, K=0.1):
    """Michaelis–Menten: Vx / (K + x)"""
    x_clamped = np.maximum(x, 0)
    return V * x_clamped / (K + x_clamped + 1e-12)


# ============================================================
#   Custom single-variable nonlinear library
#   (No interactions here — handled separately by linear part)
# ============================================================

class LGLibrary(ps.CustomLibrary):
    def __init__(self):
        super().__init__(
            library_functions=[
                lambda x: x,
                lambda x: x**2,
                hill_act_vec,
                hill_inh_vec,
                mm_vec,
            ],
            function_names=[
                "x",
                "x^2",
                "hill_act",
                "hill_inh",
                "mm",
            ],
        )


# ============================================================
#   Build SINDy model
# ============================================================

def build_sindy_model(threshold=0.01, smooth=True):
    if ps is None:
        raise RuntimeError("PySINDy not installed. Run: pip install pysindy")

    # Linear + interaction terms (captures cross-gene effects)
    library_linear = ps.PolynomialLibrary(degree=1, include_interaction=True)

    # LG nonlinearities (univariate)
    library_lg = LGLibrary()

    # Combined library: [linear+interactions] + [LG nonlinearities]
    full_library = ps.GeneralizedLibrary([library_linear, library_lg])

    differentiation = ps.SmoothedFiniteDifference(
        smoother_kws={"window_length": 11, "polyorder": 3, "axis": 0}
    )

    optimizer = ps.STLSQ(threshold=threshold)

    model = ps.SINDy(
        feature_library=full_library,
        differentiation_method=differentiation,
        optimizer=optimizer,
    )

    return model


# ============================================================
#   SINDy fitting wrapper
# ============================================================

def sindy_fit(X, t, threshold=0.01, smooth=True):
    """
    Fit SINDy with LG-style nonlinearities on time series X(t).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Time series (e.g., principal components).
    t : array-like, shape (n_samples,)
        Time stamps (can be non-uniform; SINDy will use diffs).
    threshold : float
        STLSQ sparsity threshold.
    smooth : bool
        If True, apply Savitzky–Golay smoothing before differentiation.
    """
    X = np.asarray(X, dtype=float)
    t = np.asarray(t, dtype=float)

    # Drop any rows that already contain NaNs
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    t = t[mask]

    # Optional smoothing
    if smooth and X.shape[0] >= 11:
        # make sure window_length is odd and <= n_samples
        win = min(11, X.shape[0] if X.shape[0] % 2 == 1 else X.shape[0] - 1)
        if win >= 5:  # need at least polyorder+2 points
            X = savgol_filter(X, window_length=win, polyorder=3, axis=0)

    # Final safety: ensure no NaN/inf going into feature library
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    model = build_sindy_model(threshold=threshold, smooth=smooth)
    model.fit(X, t=t)
    return model
