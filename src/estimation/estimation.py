"""
Parameter estimation scaffolding and utilities.
"""

import numpy as np
from typing import Sequence
from scipy.optimize import differential_evolution, least_squares
from scipy.integrate import solve_ivp

from src.models.leloup_goldbeter import f, default_initial_conditions, LGParams


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error between two arrays.

    Both arrays should have the same shape; use flattened/masked arrays for partial comparisons.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def simulate_model(params: LGParams, y0: Sequence[float], t_eval: np.ndarray):
    sol = solve_ivp(lambda t, y: f(t, y, params), (t_eval[0], t_eval[-1]), y0, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y

def theta_to_params(theta: np.ndarray, template: LGParams | None = None) -> LGParams:
    if template is None:
        p = LGParams()
    else:
        p = LGParams(**template.__dict__)

    # map common transcription/degradation param subset
    names = [
        "v_sP", "v_sC", "v_sB", "v_sR",
        "k_degM_P", "k_degM_C", "k_degM_B", "k_degM_R",
    ]
    for i, n in enumerate(names):
        if i < len(theta):
            setattr(p, n, float(theta[i]))
    return p


def residuals_for_theta(theta: np.ndarray, t_obs: np.ndarray, y_obs: np.ndarray,
                        param_template: LGParams | None = None,
                        observed_states: Sequence[int] | None = None,
                        y0: Sequence[float] | None = None) -> np.ndarray:
    """Return flattened residuals between observed timecourses and model predictions.

    theta -> LGParams is performed using `theta_to_params`; the ODE is simulated on
    `t_obs` and the residual is computed only on the supplied `observed_states` (a list
    of state indices from the model). If `observed_states` is None we assume the rows of
    `y_obs` match the first rows of the model output.
    """
    # normalise inputs
    t_obs = np.asarray(t_obs)
    y_obs = np.asarray(y_obs)
    if y_obs.ndim == 2 and y_obs.shape[0] == t_obs.size:
        # shape (n_times, n_genes) -> transpose to (n_genes, n_times)
        y_obs = y_obs.T

    # Map theta to params
    params = theta_to_params(np.asarray(theta, dtype=float), param_template)
    if y0 is None:
        y0 = default_initial_conditions(n_states=19)

    y_pred = simulate_model(params, y0, t_obs)  # shape (n_states, n_times)

    # which states to compare? default: first n rows of y_pred
    if observed_states is None:
        if y_obs.shape[0] <= y_pred.shape[0]:
            observed_states = list(range(y_obs.shape[0]))
        else:
            raise ValueError("y_obs contains more states than the model. Provide observed_states.")

    y_pred_sel = y_pred[observed_states, :]

    # ensure shapes match
    if y_pred_sel.shape != y_obs.shape:
        raise ValueError(f"Observed data shape {y_obs.shape} does not match prediction {y_pred_sel.shape}")

    return (y_pred_sel - y_obs).ravel()


def fit_local(theta0, bounds, data, param_template: LGParams | None = None,
              observed_states: Sequence[int] | None = None, y0=None, **kwargs):
    """Local parameter fit using scipy.least_squares.

    Parameters
    - theta0: initial guess for parameter vector (1D)
    - bounds: bounds passed to least_squares (low, high) or None for no bounds
    - data: tuple (t_obs, y_obs) where t_obs is 1D time points and y_obs is array shape (n_states, n_times)
    - param_template: LGParams dataclass used as template for unmapped params
    - observed_states: if provided, indices of model states that correspond to the rows in y_obs

    Returns the scipy OptimizeResult from least_squares.
    """
    t_obs, y_obs = data
    theta0 = np.asarray(theta0, dtype=float)

    # convert bounds to arrays accepted by least_squares
    if bounds is None:
        lb = -np.inf * np.ones_like(theta0)
        ub = np.inf * np.ones_like(theta0)
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 2 and np.array(bounds[0]).shape == theta0.shape:
        lb, ub = bounds
    else:
        # allow passing shape (n_params, 2) list of pairs
        lb = np.array([b[0] for b in bounds]) if bounds is not None else -np.inf * np.ones_like(theta0)
        ub = np.array([b[1] for b in bounds]) if bounds is not None else np.inf * np.ones_like(theta0)

    fun = lambda th: residuals_for_theta(th, t_obs, y_obs, param_template, observed_states, y0)

    res = least_squares(fun, theta0, bounds=(lb, ub))
    # add human-friendly RMSE to the result
    res.rmse = rmse(y_obs.ravel(), (res.fun + y_obs.ravel()))
    return res


def fit_global(bounds, data, param_template: LGParams | None = None,
               observed_states: Sequence[int] | None = None, y0=None, maxiter: int = 20,
               popsize: int = 15):
    """Global fit (differential evolution) using scalar objective = RMSE.

    Bounds should be a list of (low, high) per parameter.
    """
    def scalar_obj(th):
        try:
            resids = residuals_for_theta(th, data[0], data[1], param_template, observed_states, y0)
            return float(np.sqrt(np.mean(resids ** 2)))
        except Exception:
            return 1e6

    res = differential_evolution(scalar_obj, bounds, maxiter=maxiter, popsize=popsize)
    # attach the RMSE
    res.rmse = scalar_obj(res.x)
    return res


def fit_for_profile(param_vector: np.ndarray, data, fixed_idx=None, fixed_val=None, **kwargs):
    """Wrapper that performs a local fit but allows one parameter to be fixed (for profile likelihood).

    `param_vector` is treated as the initial guess; if `fixed_idx` is not None, bounds will be
    adjusted to fix that parameter at `fixed_val` during the least-squares call.
    Returns the objective (RMSE) for the best-fit under the constraint.
    """
    theta0 = np.asarray(param_vector, dtype=float)
    n = theta0.size
    # set up default bounds
    lb = -np.inf * np.ones(n)
    ub = np.inf * np.ones(n)
    if fixed_idx is not None:
        lb[fixed_idx] = float(fixed_val)
        ub[fixed_idx] = float(fixed_val)

    res = fit_local(theta0, (lb, ub), data, **kwargs)
    return float(res.rmse)
