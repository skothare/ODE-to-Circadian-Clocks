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
                       bounds, range_factor=0.5, num_points=15):
    """
    Compute profile likelihood for a single parameter.
    
    Args:
        param_name: Name of parameter (for plotting)
        param_index: Index in theta vector
        theta_opt: Optimal parameter vector
        range_factor: How far to scan (+/- range_factor around optimal)
        num_points: Number of points to scan
    """
    print(f"\nComputing profile for {param_name} (theta[{param_index}])...")
    
    opt_val = theta_opt[param_index]
    # Scan range (log scale if parameter is strictly positive usually better, but linear for now)
    # Using linear range around optimum
    scan_vals = np.linspace(opt_val * (1 - range_factor), opt_val * (1 + range_factor), num_points)
    
    # Store results
    profiles = []
    
    # Base setup
    p0 = LGParams()
    y0 = default_initial_conditions()
    observed_states = [0, 3, 6, 9]
    
    # Wrapper for optimization of OTHER parameters
    # We fix theta[param_index] = fixed_val, optimize others
    
    def constrained_loss(theta_subset, fixed_val, p_idx):
        # Reconstruct full theta
        # theta_subset has size N-1
        theta_full = np.insert(theta_subset, p_idx, fixed_val)
        return residuals_for_theta(theta_full, t_obs, y_obs, p0, observed_states, y0)

    # Initial guess for other parameters is the optimal vector minus the fixed one
    theta_others_opt = np.delete(theta_opt, param_index)

    best_chi2 = residuals_for_theta(theta_opt, t_obs, y_obs, p0, observed_states, y0)**2
    threshold = best_chi2 + 3.84 # 95% CI for 1 DOF (chi-square distribution)
    
    # Sort scan values relative to the optimum for continuation
    # We scan outwards from the optimum to keep the guess valid
    scan_vals = np.sort(scan_vals)
    opt_idx = np.searchsorted(scan_vals, opt_val)
    
    # Split into left (descending from opt) and right (ascending from opt)
    left_scan = scan_vals[:opt_idx][::-1]
    right_scan = scan_vals[opt_idx:]
    
    # Helper to run scan
    def run_scan(vals, initial_guess):
        results_map = {}
        current_guess = initial_guess.copy()
        
        for val in vals:
            # Local optimization for others
            # Increased maxiter to 500 for better convergence
            res = minimize(constrained_loss, current_guess, args=(val, param_index),
                           method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            cost = res.fun**2 
            results_map[val] = cost
            # Update guess for next step (continuation)
            current_guess = res.x
            print(f"  val={val:.4f}, cost={cost:.4f}")
        return results_map

    # Run both sides
    print(f"  Scanning right...")
    res_right = run_scan(right_scan, theta_others_opt)
    
    print(f"  Scanning left...")
    res_left = run_scan(left_scan, theta_others_opt)
    
    # Combine
    results_map = {**res_right, **res_left}
    sorted_vals = np.array(sorted(results_map.keys()))
    costs = np.array([results_map[v] for v in sorted_vals])
    
    # For plotting, use sorted_vals and costs
    scan_vals = sorted_vals
        
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(scan_vals, costs, 'b.-')
    plt.axhline(threshold, color='r', linestyle='--', label='95% threshold')
    plt.plot(opt_val, best_chi2, 'ro', label='Optimum')
    plt.xlabel(param_name)
    plt.ylabel('Objective Function (Chi-squared)')
    plt.title(f'Profile Likelihood: {param_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../figures/qc_preprocessing/profile_{param_name}.png')
    
    # Check identifiability
    # If it crosses threshold on both sides -> Identifiable
    # If flat or doesn't cross -> Non-identifiable
    min_cost = np.min(costs)
    crossings = np.where(np.diff(np.sign(np.array(costs) - threshold)))[0]
    
    status = "Identifiable"
    if len(crossings) < 2:
        if costs[0] < threshold or costs[-1] < threshold:
             status = "Practically Non-Identifiable"
             
    print(f"  Status: {status}")
    return scan_vals, costs, status

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
