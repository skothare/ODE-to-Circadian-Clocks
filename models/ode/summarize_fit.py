import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Setup paths
ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from models.leloup_goldbeter import f, default_initial_conditions, LGParams
from estimation.estimation import theta_to_params

print("="*60)
print("FIT QUALITY SUMMARY")
print("="*60)

# Load data
meta = pd.read_csv('../data/processed/sample_metadata.csv')
expr = pd.read_csv('../data/processed/expression_matrix.csv', index_col=0)
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
    'M_P': mp_raw,
    'M_C': mc_raw,
    'M_B': mb_raw,
    'M_R': mr_raw,
    't_idx': [gsm_to_time[g] for g in mp_raw.index]
})

data_mean = data_df.groupby('t_idx').mean().sort_index()
t_obs = data_mean.index.values
y_obs_raw = data_mean[['M_P', 'M_C', 'M_B', 'M_R']].values.T

# Load optimal parameters (you'll need to paste these from the output)
# For now, let's use defaults and note that user should update
p0 = LGParams()
# TODO: Replace with actual optimized values from the fit
# theta_opt = np.array([...])  # From run output
# p_opt = theta_to_params(theta_opt, p0)
p_opt = p0  # Using defaults for now

y0 = default_initial_conditions()
observed_states = [0, 3, 6, 9]

# Simulate to limit cycle
t_spinup = 500
t_window = t_obs[-1] - t_obs[0] + 24
t_total = t_spinup + t_window
t_eval = np.linspace(0, t_total, 2000)

sol = solve_ivp(lambda t, y: f(t, y, p_opt), (0, t_total), y0, t_eval=t_eval, method='LSODA')
mask = t_eval > t_spinup
t_steady = t_eval[mask] - t_spinup
y_steady = sol.y[:, mask]
y_model_subset = y_steady[observed_states, :]

labels = ['Per mRNA', 'Cry mRNA', 'Bmal1 mRNA', 'Rev-Erb mRNA']
shifts = np.linspace(0, 24, 100)

print("\nPer-Gene Fit Quality:")
print("-" * 60)

overall_r2 = []
overall_rmse = []

for i, idx in enumerate(observed_states):
    row_obs = y_obs_raw[i, :]
    row_model = y_model_subset[i, :]
    
    model_func = interp1d(t_steady, row_model, kind='cubic', fill_value="extrapolate")
    
    best_phi = 0
    best_r2 = -np.inf
    best_rmse = np.inf
    
    for phi in shifts:
        pred = model_func(t_obs + phi)
        
        # Scale to data space
        mu_m = np.mean(pred)
        sig_m = np.std(pred)
        mu_d = np.mean(row_obs)
        sig_d = np.std(row_obs)
        
        a = sig_d / (sig_m + 1e-9)
        b = mu_d - a * mu_m
        pred_scaled = a * pred + b
        
        # Calculate metrics
        ss_res = np.sum((row_obs - pred_scaled)**2)
        ss_tot = np.sum((row_obs - np.mean(row_obs))**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((row_obs - pred_scaled)**2))
        
        if r2 > best_r2:
            best_r2 = r2
            best_rmse = rmse
            best_phi = phi
    
    overall_r2.append(best_r2)
    overall_rmse.append(best_rmse)
    
    print(f"{labels[i]:20s} | Phase Shift: {best_phi:5.2f}h | R²: {best_r2:6.4f} | RMSE: {best_rmse:8.2f}")

print("-" * 60)
print(f"{'OVERALL':20s} | {'':13s} | R²: {np.mean(overall_r2):6.4f} | RMSE: {np.mean(overall_rmse):8.2f}")
print("="*60)

print("\nNOTE: These metrics use default parameters.")
print("Run 02_estimation.py to get optimized fit, then update theta_opt in this script.")
