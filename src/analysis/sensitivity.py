"""
Local parameter sensitivity via finite differences (forward difference) for a set of parameters.
"""
import numpy as np
from scipy.integrate import solve_ivp
from src.models.leloup_goldbeter import f


def finite_difference_sensitivities(param_vector, param_map, y0, t_eval, eps=1e-6):
    base_params = param_map(param_vector)
    sol0 = solve_ivp(lambda t, y: f(t, y, base_params), (t_eval[0], t_eval[-1]), y0, t_eval=t_eval)
    y0sol = sol0.y

    n_params = param_vector.size
    n_states, n_times = y0sol.shape
    S = np.zeros((n_states, n_times, n_params))

    for i in range(n_params):
        pv = param_vector.copy()
        pv[i] += eps
        params_plus = param_map(pv)
        solp = solve_ivp(lambda t, y: f(t, y, params_plus), (t_eval[0], t_eval[-1]), y0, t_eval=t_eval)
        y_plus = solp.y
        S[:, :, i] = (y_plus - y0sol) / eps

    return S
