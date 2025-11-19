import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from src.models.leloup_goldbeter import f, LGParams, default_initial_conditions


def test_basic_integration_and_positivity():
    params = LGParams()
    y0 = default_initial_conditions()
    t_span = (0.0, 240.0)  # simulate ~10 days
    t_eval = np.linspace(t_span[0], t_span[1], 2000)

    sol = solve_ivp(lambda t, y: f(t, y, params), t_span=t_span, y0=y0, t_eval=t_eval)
    assert sol.success

    # Positivity: all states should stay >= 0
    assert np.all(sol.y >= -1e-8)

    # Check number of states (19 equations)
    assert sol.y.shape[0] == 19

    # Basic oscillatory check: variance of one variable should be non-trivial
    var_M = np.var(sol.y[0])
    assert var_M > 1e-6

    # Check for approximate period using crude peak detection
    peaks, _ = find_peaks(sol.y[1], height=np.mean(sol.y[1]))
    # If there are peaks, check if distances average around 24 (not strict)
    if len(peaks) > 3:
        periods = np.diff(sol.t[peaks])
        mean_period = np.mean(periods)
        # period should be in plausible circadian range
        assert 15.0 < mean_period < 35.0