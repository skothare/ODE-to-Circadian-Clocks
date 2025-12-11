
import pandas as pd
import numpy as np
import re
import gzip
from pathlib import Path
import os

def parse_meta(fn: Path):
    """
    Parses metadata from the filename of an Agilent FE file.
    Expected format: GSM1168586_BB0012_R_1.txt.gz
    """
    name = fn.stem  # removes .gz
    if name.endswith(".txt"):
        name = name[:-4]
    
    # Regex to match GSM, Subject, Condition (R/S), Time Index
    m1 = re.match(r"(GSM\d+)_([A-Za-z0-9]+)_([RS])_(\d+)$", name)
    if not m1:
        return None
        
    gsm, subj, cond, t_idx = m1.groups()
    return dict(gsm=gsm, subject=subj, condition=cond, t_idx=int(t_idx), file=str(fn))

def load_expression_data(raw_dir: Path, processed_dir: Path):
    """
    Loads raw Agilent FE files, extracts 'gProcessedSignal', and aggregates into a matrix.
    """
    files = sorted(raw_dir.glob("*.txt.gz"))
    if not files:
        print(f"No .txt.gz files found in {raw_dir}")
        return

    # 1. Parse Metadata
    meta_list = []
    for f in files:
        m = parse_meta(f)
        if m:
            meta_list.append(m)
    
    meta = pd.DataFrame(meta_list).sort_values(["subject","condition","t_idx"])
    meta.to_csv(processed_dir / "sample_metadata.csv", index=False)
    print(f"Saved metadata for {len(meta)} samples to {processed_dir / 'sample_metadata.csv'}")

    # 2. Extract Expression
    # We will read each file, take 'GeneName' and 'gProcessedSignal'.
    # Note: Using a simplified approach to avoid memory issues with 288 files at once if possible,
    # but for 30k genes x 288 samples it fits in memory.
    
    all_series = []
    gene_names = None
    
    print("Loading expression data from files...")
    for idx, row in meta.iterrows():
        fn = Path(row['file'])
        # Read only relevant columns
        # Filter for 'gIsWellAboveBG' if desired? User used 'gProcessedSignal' in datapeek.
        # We will use 'gProcessedSignal' as the primary value.
        
        try:
            # Skip header lines usually 9, read data
            # Datapeek used pd.read_csv with compression
            df = pd.read_csv(fn, sep="\t", compression="gzip", comment='#', low_memory=False)
            
            # Filter control probes if needed? 
            # Usually we filter 'ControlType' == 0
            if 'ControlType' in df.columns:
                df = df[df['ControlType'] == 0]
            
            # Group by GeneName and take mean (handle duplicates)
            # datapeek didn't show explicit aggregation but it's good practice
            if 'GeneName' not in df.columns:
                print(f"Warning: GeneName missing in {fn.name}")
                continue
                
            sample_expr = df.groupby('GeneName')['gProcessedSignal'].mean()
            
            if gene_names is None:
                gene_names = sample_expr.index
            
            # Reindex to ensure alignment
            sample_expr = sample_expr.reindex(gene_names)
            sample_expr.name = row['gsm']
            all_series.append(sample_expr)
            
            if idx % 20 == 0:
                print(f"Processed {idx}/{len(meta)}")
                
        except Exception as e:
            print(f"Error processing {fn.name}: {e}")

    # 3. Create Matrix
    expr_matrix = pd.concat(all_series, axis=1)
    expr_matrix.to_csv(processed_dir / "expression_matrix.csv")
    print(f"Saved expression matrix ({expr_matrix.shape}) to {processed_dir / 'expression_matrix.csv'}")

if __name__ == "__main__":
    # Define paths relative to this script or project root
    # Assuming script is in ROOT/preprocessing/
    # ROOT is parent of currently script directory if run from there
    
    # Adjust based on run location. We assume run from project root usually.
    BASE_DIR = Path.cwd() 
    if (BASE_DIR / 'data').exists():
        RAW_DIR = BASE_DIR / "data/raw"
        PROCESSED_DIR = BASE_DIR / "data/processed"
    else:
        # Fallback
        RAW_DIR = Path("../data/raw")
        PROCESSED_DIR = Path("../data/processed")
        
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    load_expression_data(RAW_DIR, PROCESSED_DIR)
