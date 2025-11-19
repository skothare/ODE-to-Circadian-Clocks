"""
Profile likelihood scaffolding
"""
import numpy as np


def profile_likelihood(param_index, param_grid, param_vector, fit_function, data):
    obj_values = []
    for val in param_grid:
        obj = fit_function(param_vector, data, fixed_idx=param_index, fixed_val=val)
        obj_values.append(obj)
    return np.array(param_grid), np.array(obj_values)


def fit_wrapper_for_profile_local(param_vector, data, fixed_idx=None, fixed_val=None, **kwargs):
    """Compatibility wrapper expected by profile_likelihood.

    It delegates to `src.estimation.estimation.fit_local` with bounds to fix one parameter
    if requested and returns the scalar objective (RMSE).
    """
    from src.estimation.estimation import fit_for_profile

    return fit_for_profile(param_vector, data, fixed_idx=fixed_idx, fixed_val=fixed_val, **kwargs)
