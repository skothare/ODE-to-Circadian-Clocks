import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Setup paths
ROOT = Path.cwd().parent.parent  # models/ode -> models -> ROOT
sys.path.insert(0, str(ROOT))

from models.ode.leloup_goldbeter import f, default_initial_conditions, LGParams
from models.ode.estimation.estimation import residuals_for_theta, theta_to_params

def profile_likelihood(param_name, param_index, theta_opt, t_obs, y_obs, 
                       bounds, range_factor=0.5, num_points=21):
    """
    Compute profile likelihood for a single parameter.
    Scans on both sides of the MLE to assess practical identifiability.
    
    Args:
        param_name: Name of parameter (for plotting)
        param_index: Index in theta vector
        theta_opt: Optimal parameter vector (initial estimate of MLE)
        range_factor: How far to scan (+/- range_factor around optimal)
        num_points: Number of points to scan (odd number ensures MLE is included)
    """
    print(f"\nComputing profile for {param_name} (theta[{param_index}])...")
    
    # Base setup
    p0 = LGParams()
    y0 = default_initial_conditions()
    observed_states = [0, 3, 6, 9]
    
    # First, compute loss at the provided optimum (this is our MLE estimate)
    mle_val = theta_opt[param_index]
    mle_loss = residuals_for_theta(theta_opt, t_obs, y_obs, p0, observed_states, y0)
    print(f"  MLE estimate: {mle_val:.4f}, loss: {mle_loss:.4f}")
    
    # Wrapper for optimization of OTHER parameters (nuisance parameters)
    def constrained_loss(theta_subset, fixed_val, p_idx):
        theta_full = np.insert(theta_subset, p_idx, fixed_val)
        return residuals_for_theta(theta_full, t_obs, y_obs, p0, observed_states, y0)

    # Initial guess for nuisance parameters
    theta_others_mle = np.delete(theta_opt, param_index)
    
    # Create scan values centered on MLE
    # Ensure we include the MLE value itself
    lower = mle_val * (1 - range_factor)
    upper = mle_val * (1 + range_factor)
    
    # Create scan points on each side of MLE
    n_each_side = num_points // 2
    left_vals = np.linspace(lower, mle_val, n_each_side + 1)[:-1]  # exclude MLE
    right_vals = np.linspace(mle_val, upper, n_each_side + 1)  # include MLE
    
    # Scan outward from MLE for better continuation
    results = {}
    
    # Add MLE point
    results[mle_val] = mle_loss
    
    # Scan RIGHT from MLE (ascending)
    print(f"  Scanning right from MLE...")
    current_guess = theta_others_mle.copy()
    for val in right_vals[1:]:  # skip MLE, already added
        res = minimize(constrained_loss, current_guess, args=(val, param_index),
                       method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        results[val] = res.fun
        current_guess = res.x  # warm start
        print(f"    val={val:.4f}, loss={res.fun:.4f}")
    
    # Scan LEFT from MLE (descending)
    print(f"  Scanning left from MLE...")
    current_guess = theta_others_mle.copy()
    for val in left_vals[::-1]:  # descend from MLE
        res = minimize(constrained_loss, current_guess, args=(val, param_index),
                       method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        results[val] = res.fun
        current_guess = res.x
        print(f"    val={val:.4f}, loss={res.fun:.4f}")
    
    # Sort results by parameter value
    scan_vals = np.array(sorted(results.keys()))
    costs = np.array([results[v] for v in scan_vals])

    
    # Compute delta NLLH: difference from minimum
    # For RMSE-based loss, convert to approximate log-likelihood scale
    # NLLH ≈ 0.5 * n * log(RSS/n) where RSS = n * RMSE^2
    # Simplified: just use squared RMSE as proxy for chi-squared
    costs_sq = costs ** 2  # Squared RMSE ~ sum of squared residuals
    min_cost_sq = np.min(costs_sq)
    delta_nllh = costs_sq - min_cost_sq  # Delta from minimum
    
    # 95% CI threshold for chi-squared with 1 DOF = 3.84
    # For profile likelihood: threshold at delta = 1.92 (half the chi-sq critical value)
    threshold_delta = 1.92
    
    # Find where the minimum occurs
    min_idx = np.argmin(costs_sq)
    min_param_val = scan_vals[min_idx]
        
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(scan_vals, delta_nllh, 'b.-', linewidth=1.5, markersize=8)
    plt.axhline(threshold_delta, color='r', linestyle='--', label='95% CI threshold')
    plt.axvline(min_param_val, color='g', linestyle=':', alpha=0.7)
    plt.scatter([min_param_val], [0], color='red', s=100, zorder=5, label=f'MLE ({min_param_val:.3f})')
    plt.xlabel(param_name)
    plt.ylabel('$\\Delta$ NLLH')
    plt.title(f'Profile Likelihood: {param_name}')
    plt.ylim(bottom=-0.1)  # Ensure minimum is visible at 0
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / f'figures/profile_{param_name}.png', dpi=150)
    plt.close()
    
    # Check identifiability based on delta NLLH
    # If profile crosses threshold on both sides of MLE -> Identifiable
    # If flat or doesn't cross -> Non-identifiable
    crossings = np.where(np.diff(np.sign(delta_nllh - threshold_delta)))[0]
    
    # Check if MLE is at or near the minimum of the profile
    mle_in_scan = np.argmin(np.abs(scan_vals - mle_val))
    
    status = "Identifiable"
    if len(crossings) < 2:
        # Doesn't cross threshold on both sides
        status = "Practically Non-Identifiable"
    elif delta_nllh[0] < threshold_delta or delta_nllh[-1] < threshold_delta:
        # Threshold not exceeded at boundaries
        status = "Practically Non-Identifiable"
             
    print(f"  Status: {status}")
    return scan_vals, delta_nllh, status

# --- Load Data and Best Parameters ---
# NOTE: This assumes we have a 'best parameters' file or we hardcode from previous run
# Since 02_estimation.py only printed them, I will placeholders here.
# USER MUST UPDATE 'theta_opt' below with values from estimation output!

theta_opt = np.array([0.45258916, 2.44051586, 0.43268779, 1.62908918, 0.73942923, 0.36399779, 0.87584874, 0.40773593])
# Update logic to read from file if needed

print("Loading real data...")
meta = pd.read_csv(ROOT / 'data/processed/sample_metadata.csv')
expr = pd.read_csv(ROOT / 'data/processed/expression_matrix.csv', index_col=0)

# Filter for condition 'R' (using full dataset for identifiability)
meta_r = meta[meta['condition'] == 'R']
gsm_to_time = dict(zip(meta_r['gsm'], meta_r['t_idx']))
valid_gsms = [g for g in expr.columns if g in gsm_to_time]
expr_r = expr[valid_gsms]

def get_gene_expr(df, genes):
    found_genes = [g for g in genes if g in df.index]
    return df.loc[found_genes].mean(axis=0)

mp_raw = get_gene_expr(expr_r, ['PER1', 'PER2', 'PER3'])
mc_raw = get_gene_expr(expr_r, ['CRY1', 'CRY2'])
mb_raw = get_gene_expr(expr_r, ['ARNTL'])
mr_raw = get_gene_expr(expr_r, ['NR1D1'])

data_df = pd.DataFrame({
    'M_P': mp_raw, 'M_C': mc_raw, 'M_B': mb_raw, 'M_R': mr_raw,
    't_idx': [gsm_to_time[g] for g in mp_raw.index]
})
data_mean = data_df.groupby('t_idx').mean().sort_index()
t_obs = data_mean.index.values
y_obs = data_mean[['M_P', 'M_C', 'M_B', 'M_R']].values.T

print("Running Profile Likelihood...")

# Names of parameters in theta
param_names = ['v_sP', 'v_sC', 'v_sB', 'v_sR', 'k_degM_P', 'k_degM_C', 'k_degM_B', 'k_degM_R']
defaults = np.array([1.5, 1.2, 1.4, 1.1, 0.23, 0.25, 0.28, 0.22]) # Default reference
bounds = [(d * 0.2, d * 5.0) for d in defaults]

# Profile the first and last parameter as examples
# (Profiling all 8 takes time)
results = {}
for i, name in enumerate([param_names[0], param_names[-1]]): # v_sP and k_degM_R
    idx = param_names.index(name)
    scan, costs, status = profile_likelihood(name, idx, theta_opt, t_obs, y_obs, bounds)
    results[name] = status

print("\nIdentifiability Summary:")
for k, v in results.items():
    print(f"{k}: {v}")
