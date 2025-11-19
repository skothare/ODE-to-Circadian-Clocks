import numpy as np
from src.analysis.sensitivity import finite_difference_sensitivities
from src.models.leloup_goldbeter import default_initial_conditions


def param_map(vec):
    from src.models.leloup_goldbeter import LGParams
    p = LGParams()
    p.v_sP = float(vec[0])
    # vary PER transcription and PER mRNA degradation (match LGParams fields)
    p.k_degM_P = float(vec[1])
    return p


def test_sensitivity_runs():
    pv = np.array([1.0, 0.2])
    y0 = default_initial_conditions()
    t_eval = np.linspace(0, 48, 100)
    S = finite_difference_sensitivities(pv, param_map, y0, t_eval)
    assert S.shape[2] == 2
    assert S.ndim == 3
