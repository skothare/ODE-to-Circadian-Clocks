import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Setup paths
ROOT = Path.cwd()
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from src.models.leloup_goldbeter import f, default_initial_conditions, LGParams
from src.estimation.estimation import simulate_model

# Load real data
print("Loading real data...")
meta = pd.read_csv('data/processed/sample_metadata.csv')
expr = pd.read_csv('data/processed/expression_matrix.csv', index_col=0)
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

# Normalize data
y_obs_norm = y_obs_raw / np.max(y_obs_raw, axis=1, keepdims=True)
y_obs = y_obs_norm * 2.0 # Scale to approx model amplitude

# Simulate default model
print("Simulating default model...")
p0 = LGParams()
y0 = default_initial_conditions()
# Simulate for longer to reach limit cycle
t_sim = np.linspace(0, 72, 200) 
try:
    X_default = simulate_model(p0, y0, t_sim)
    
    # Plot
    plt.figure(figsize=(12, 8))
    labels = ['Per mRNA', 'Cry mRNA', 'Bmal1 mRNA', 'Rev-Erb mRNA']
    observed_states = [0, 3, 6, 9]
    
    for i, idx in enumerate(observed_states):
        plt.subplot(2, 2, i+1)
        plt.plot(t_obs, y_obs[i], 'o', label='Observed (Scaled)')
        plt.plot(t_sim, X_default[idx], '-', label='Default Model')
        plt.title(labels[i])
        plt.legend()
    plt.tight_layout()
    plt.savefig('figures/qc_preprocessing/debug_model_default.png')
    print("Saved debug plot to figures/qc_preprocessing/debug_model_default.png")
    
    # Check period/amplitude
    print("Model max values:", np.max(X_default[observed_states], axis=1))
    print("Model min values:", np.min(X_default[observed_states], axis=1))
    
except Exception as e:
    print(f"Simulation failed: {e}")
