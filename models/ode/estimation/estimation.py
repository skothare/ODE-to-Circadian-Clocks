"""
Parameter estimation scaffolding and utilities.
"""

import numpy as np
from typing import Sequence
from scipy.optimize import differential_evolution, least_squares
from scipy.integrate import solve_ivp

from models.ode.leloup_goldbeter import f, default_initial_conditions, LGParams


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error between two arrays.

    Both arrays should have the same shape; use flattened/masked arrays for partial comparisons.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def simulate_model(params: LGParams, y0: Sequence[float], t_eval: np.ndarray):
    sol = solve_ivp(lambda t, y: f(t, y, params), (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='LSODA')
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


from scipy.interpolate import interp1d

def residuals_for_theta(theta: np.ndarray, t_obs: np.ndarray, y_obs: np.ndarray,
                        param_template: LGParams | None = None,
                        observed_states: Sequence[int] | None = None,
                        y0: Sequence[float] | None = None) -> float:
    """Return scalar loss (RMSE-like) for a proposed theta using limit cycle matching.
    
    Simulates the model to steady state (limit cycle), then finds the best phase shift
    and compares Z-scores of the model and data.
    """
    # 1. Update Parameters
    params = theta_to_params(theta, param_template)
    
    if y0 is None:
        y0 = default_initial_conditions(n_states=19)

    # 2. SPIN UP: Simulate long enough to reach limit cycle (e.g., 500h)
    # We simulate extra time to cover the data window after spin-up
    t_spinup = 500
    t_window = t_obs[-1] - t_obs[0] + 24 # Window size + extra day for shifting
    t_total = t_spinup + t_window
    
    try:
        # Simulate on a dense grid for accurate interpolation
        t_eval = np.linspace(0, t_total, 2000)
        # Use LSODA for stiffness
        sol = solve_ivp(lambda t, y: f(t, y, params), (0, t_total), y0, t_eval=t_eval, method='LSODA')
        if not sol.success:
             return 1e6
    except RuntimeError:
        return 1e6 # Penalty for failure

    # 3. EXTRACT LIMIT CYCLE (Last part of simulation)
    # We only care about the data after t=500
    mask = t_eval > t_spinup
    t_steady = t_eval[mask] - t_spinup # Reset time to 0 relative to spinup
    y_steady = sol.y[:, mask]
    
    # 4. PHASE MATCHING via Cross-Correlation (or simple sliding)
    # Instead of optimizing 'phi' as a parameter (which is hard), 
    # we can calculate residuals at the BEST phase shift for this gene.
    
    residuals = []
    
    # which states to compare? default: first n rows of y_pred
    if observed_states is None:
        if y_obs.shape[0] <= y_steady.shape[0]:
            observed_states = list(range(y_obs.shape[0]))
        else:
            raise ValueError("y_obs contains more states than the model. Provide observed_states.")

    # Extract just the relevant model states
    y_model_subset = y_steady[observed_states, :] # Shape (n_states, n_points)
    
    # Ensure y_obs is (n_states, n_times)
    if y_obs.ndim == 2 and y_obs.shape[1] != t_obs.size and y_obs.shape[0] == t_obs.size:
         y_obs = y_obs.T

    for i, row_obs in enumerate(y_obs): # For each gene (Per, Cry, etc.)
        row_model = y_model_subset[i, :]
        
        # Create interpolator for the model curve
        # We wrap the time to handle the cycle (optional, but linear interp is safer here)
        model_func = interp1d(t_steady, row_model, kind='cubic', fill_value="extrapolate")
        
        # Try a few phase shifts (0h to 24h) to find the best alignment
        shifts = np.linspace(0, 24, 24) 
        best_gene_resid = np.inf
        
        for phi in shifts:
            # Get model prediction at observed times + phase shift
            pred = model_func(t_obs + phi)
            
            # 5. SCALE-FREE COMPARISON (Z-Score)
            # Normalize both to Mean=0, Std=1. 
            # This forces the optimizer to match SHAPE and PERIOD, not Amplitude.
            pred_z = (pred - np.mean(pred)) / (np.std(pred) + 1e-9)
            obs_z  = (row_obs - np.mean(row_obs)) / (np.std(row_obs) + 1e-9)
            
            res = np.sum((pred_z - obs_z)**2)
            if res < best_gene_resid:
                best_gene_resid = res
                
        residuals.append(best_gene_resid)

    # Return scalar sum of squares (for differential_evolution)
    return np.sqrt(np.sum(residuals))


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
        return residuals_for_theta(th, data[0], data[1], param_template, observed_states, y0)

    res = differential_evolution(scalar_obj, bounds, maxiter=maxiter, popsize=popsize)
    # attach the RMSE
    res.rmse = res.fun
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
