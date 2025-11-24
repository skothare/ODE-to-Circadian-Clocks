import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gzip

# Setup paths
ROOT = Path.cwd()
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

# Import from src
from preprocessing.gse48113 import parse_meta, read_agilent_fe, extract_expression, quantile_normalize

# Data directories
RAW_DIR = Path('data/raw')
# Check if data is in data/raw or just data/
# The user said "data is downloaded and stored in data/".
# But the notebook uses 'data/raw'.
# Let's check where the .txt.gz files are.
# In step 11, list_dir of 'data' showed .txt.gz files directly in 'data'.
# So RAW_DIR should be 'data'.
RAW_DIR = Path('data')
PROCESSED_DIR = Path('data/processed')
FIG_DIR = Path('figures/qc_preprocessing')

# Ensure dirs exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Looking for .txt.gz files in {RAW_DIR}")
files = sorted(RAW_DIR.glob("*.txt.gz"))
if not files:
    raise RuntimeError(f"No .txt.gz files found in {RAW_DIR}")

print(f"Found {len(files)} files.")

# Parse metadata
print("Parsing metadata...")
meta = pd.DataFrame([parse_meta(f) for f in files]).sort_values(["subject", "condition", "t_idx"])
meta.to_csv(PROCESSED_DIR / "sample_metadata.csv", index=False)
print("Number of samples:", len(meta))
print(meta.head())

# Quick QC plot
print("Generating sample counts plot...")
plt.figure(figsize=(6, 4))
sns.countplot(data=meta, x="t_idx", hue="condition")
plt.title("Sample counts by time index and condition")
plt.tight_layout()
plt.savefig(FIG_DIR / "sample_counts_by_time_condition.png", dpi=200)
plt.close()

# Example FE file processing
if len(files) > 0:
    print("Processing example FE file...")
    df0 = read_agilent_fe(files[0])
    print('Example FE file shape:', df0.shape)
    print('Example FE columns:', df0.columns[:15].tolist())
    ex0 = extract_expression(df0)
    print('Extracted expression example:')
    print(ex0.head())

    plt.figure(figsize=(5, 4))
    sns.histplot(ex0['intensity'].astype(float), bins=80, log_scale=(False, True))
    plt.xlabel('Raw intensity (gProcessedSignal)')
    plt.ylabel('Count (log scale)')
    plt.title('Raw intensity distribution (single array)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'raw_intensity_hist_single_array.png', dpi=200)
    plt.close()

# Process all files and save expression data
print("Processing all FE files to build expression matrix...")
dfs = []
for i, row in meta.iterrows():
    fpath = row['file']
    # print(f"Processing {fpath}...")
    try:
        df_fe = read_agilent_fe(Path(fpath))
        df_ex = extract_expression(df_fe)
        # Add metadata columns
        df_ex['gsm'] = row['gsm']
        df_ex['subject'] = row['subject']
        df_ex['condition'] = row['condition']
        df_ex['t_idx'] = row['t_idx']
        dfs.append(df_ex)
    except Exception as e:
        print(f"Error processing {fpath}: {e}")

if dfs:
    print("Concatenating dataframes...")
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Save full tidy dataframe (might be large)
    # full_df.to_csv(PROCESSED_DIR / "expression_tidy.csv", index=False)
    
    # Pivot to create a matrix: Genes x Samples (or Time/Condition)
    # We need to aggregate probes to genes first.
    # Let's take the mean intensity for each gene in each sample.
    print("Aggregating by gene and sample...")
    # Group by gene, condition, t_idx, subject
    # We want a time series for each condition/subject.
    
    # First, ensure 'intensity' is numeric
    full_df['intensity'] = pd.to_numeric(full_df['intensity'], errors='coerce')
    
    # Aggregate probes for the same gene in the same sample
    # (Assuming multiple probes per gene)
    # We'll group by GSM (sample) and Gene
    expr_matrix = full_df.groupby(['gsm', 'gene'])['intensity'].mean().reset_index()
    
    # Pivot: Rows = Genes, Cols = GSM
    expr_matrix_wide = expr_matrix.pivot(index='gene', columns='gsm', values='intensity')
    
    output_path = PROCESSED_DIR / "expression_matrix.csv"
    expr_matrix_wide.to_csv(output_path)
    print(f"Saved expression matrix to {output_path}")
    print("Matrix shape:", expr_matrix_wide.shape)
else:
    print("No data processed.")

print("Script 01 completed successfully.")
