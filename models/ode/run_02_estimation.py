import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# Setup paths
# Setup paths
ROOT = Path.cwd().parent.parent  # models/ode -> models -> ROOT
sys.path.insert(0, str(ROOT))

from models.ode.leloup_goldbeter import f, default_initial_conditions, LGParams
from models.ode.estimation.estimation import residuals_for_theta, theta_to_params, fit_global, simulate_model

print("Loading real data...")
# Load metadata and expression matrix
meta = pd.read_csv(ROOT / 'data/processed/sample_metadata.csv')
expr = pd.read_csv(ROOT / 'data/processed/expression_matrix.csv', index_col=0)

# Filter for condition 'R'
meta_r = meta[meta['condition'] == 'R']
gsm_to_time = dict(zip(meta_r['gsm'], meta_r['t_idx']))
valid_gsms = [g for g in expr.columns if g in gsm_to_time]
expr_r = expr[valid_gsms]

# Helper to get mean expression
def get_gene_expr(df, genes):
    found_genes = [g for g in genes if g in df.index]
    if not found_genes:
        raise ValueError(f"None of {genes} found in expression matrix")
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

# NO SCALING - Pass raw data (or log-transformed if needed, but raw is fine for Z-score)
y_obs = y_obs_raw
# Maybe log transform? Gene expression is often log-normal.
# The user said "Pass the raw (or log-transformed) data".
# Let's stick to raw for now, Z-score handles mean/std.
# But if variance increases with mean, log is better.
# Let's use raw as per user's primary suggestion.

print(f"Data prepared. Time points: {len(t_obs)}")

# Model Setup
p0 = LGParams()
y0 = default_initial_conditions()
observed_states = [0, 3, 6, 9]

# Define bounds for global optimization
defaults = np.array([p0.v_sP, p0.v_sC, p0.v_sB, p0.v_sR,
                     p0.k_degM_P, p0.k_degM_C, p0.k_degM_B, p0.k_degM_R])
bounds = [(d * 0.2, d * 5.0) for d in defaults]

print("Starting global fit (Limit Cycle + Phase Matching)...")
print("This may take a few minutes...")

try:
    # Use fit_global
    res = fit_global(bounds, (t_obs, y_obs), param_template=p0, observed_states=observed_states, 
                     y0=y0, maxiter=50, popsize=20)
    
    print('Global fit RMSE (Z-score space):', res.rmse)
    print('Optimized theta:', res.x)
    
    # --- POST PROCESSING & PLOTTING ---
    p_opt = theta_to_params(res.x, p0)
    
    # 1. Spin up to limit cycle
    t_spinup = 500
    t_window = t_obs[-1] - t_obs[0] + 24
    t_total = t_spinup + t_window
    t_eval = np.linspace(0, t_total, 2000)
    
    sol = solve_ivp(lambda t, y: f(t, y, p_opt), (0, t_total), y0, t_eval=t_eval, method='LSODA')
    if not sol.success:
        raise RuntimeError("Final simulation failed")
        
    mask = t_eval > t_spinup
    t_steady = t_eval[mask] - t_spinup
    y_steady = sol.y[:, mask]
    y_model_subset = y_steady[observed_states, :]
    
    # 2. Find best phase shift and scaling for plotting
    plt.figure(figsize=(12, 8))
    labels = ['Per mRNA', 'Cry mRNA', 'Bmal1 mRNA', 'Rev-Erb mRNA']
    
    shifts = np.linspace(0, 24, 100) # Finer grid for plotting
    
    for i, idx in enumerate(observed_states):
        row_obs = y_obs[i, :]
        row_model = y_model_subset[i, :]
        
        model_func = interp1d(t_steady, row_model, kind='cubic', fill_value="extrapolate")
        
        best_phi = 0
        best_resid = np.inf
        
        # Find best phi (re-doing the search to get the value)
        for phi in shifts:
            pred = model_func(t_obs + phi)
            pred_z = (pred - np.mean(pred)) / (np.std(pred) + 1e-9)
            obs_z  = (row_obs - np.mean(row_obs)) / (np.std(row_obs) + 1e-9)
            res_val = np.sum((pred_z - obs_z)**2)
            if res_val < best_resid:
                best_resid = res_val
                best_phi = phi
        
        print(f"{labels[i]}: Best phase shift = {best_phi:.2f} h")
        
        # 3. Calculate Scaling (a, b) to map Model -> Data
        # y_obs approx a * y_model + b
        # We want to plot the model on top of the data.
        # Let's map model to data space.
        # y_model_aligned = model_func(t_plot + best_phi)
        # We matched Z-scores: (y_m - mu_m)/sig_m = (y_d - mu_d)/sig_d
        # y_d = (sig_d/sig_m) * y_m + (mu_d - (sig_d/sig_m)*mu_m)
        
        # Get model stats on the observed time points (for fair comparison of mean/std)
        pred_at_obs = model_func(t_obs + best_phi)
        mu_m = np.mean(pred_at_obs)
        sig_m = np.std(pred_at_obs)
        
        mu_d = np.mean(row_obs)
        sig_d = np.std(row_obs)
        
        a = sig_d / (sig_m + 1e-9)
        b = mu_d - a * mu_m
        
        # Plotting
        plt.subplot(2, 2, i+1)
        plt.plot(t_obs, row_obs, 'o', label='Observed')
        
        # Smooth model curve
        t_plot = np.linspace(t_obs[0], t_obs[-1], 200)
        y_plot_raw = model_func(t_plot + best_phi)
        y_plot_scaled = a * y_plot_raw + b
        
        plt.plot(t_plot, y_plot_scaled, '-', label=f'Fitted (Shift={best_phi:.1f}h)')
        plt.title(labels[i])
        plt.legend()
        
        # Calculate R² and RMSE
        pred_scaled_at_obs = a * pred_at_obs + b
        ss_res = np.sum((row_obs - pred_scaled_at_obs)**2)
        ss_tot = np.sum((row_obs - np.mean(row_obs))**2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((row_obs - pred_scaled_at_obs)**2))
        
        print(f"{labels[i]:20s} | Phase Shift: {best_phi:5.2f}h | R²: {r2:6.4f} | RMSE: {rmse:8.2f}")
        
    plt.tight_layout()
    plt.savefig('figures/qc_preprocessing/estimation_fit_real_data_refined.png')
    print("Saved fit plot to figures/qc_preprocessing/estimation_fit_real_data_refined.png")

except Exception as e:
    print(f"Estimation failed: {e}")
    import traceback
    traceback.print_exc()

print("Script 02 completed successfully.")
