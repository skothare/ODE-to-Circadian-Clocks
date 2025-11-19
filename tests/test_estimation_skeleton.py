import numpy as np
from src.estimation.estimation import rmse, simulate_model
from src.models.leloup_goldbeter import LGParams, default_initial_conditions


def test_rmse_and_sim():
    params = LGParams()
    y0 = default_initial_conditions()
    t_eval = np.linspace(0, 48, 200)
    y = simulate_model(params, y0, t_eval)
    # default implementation now returns the 19-state Leloup-Goldbeter model; check shape and types
    assert y.shape[0] == 19
    assert y.shape[1] == t_eval.size

    true = np.zeros_like(y[0])
    ls = rmse(true, y[0])
    assert ls >= 0
