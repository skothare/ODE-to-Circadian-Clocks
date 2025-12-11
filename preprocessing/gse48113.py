"""
Helper functions for preprocessing GSE48113 Agilent FE files.
These are thin wrappers and pure-Python helpers that can be used from the notebooks.
"""
from pathlib import Path
import gzip
import re
import pandas as pd
import numpy as np

INT_CHOICES = ["gProcessedSignal", "gMeanSignal", "ProcessedSignal", "Signal"]
PROBE_KEYS = ["ProbeName", "SystematicName", "FeatureNum"]
GENE_KEYS = ["GeneName", "Gene Symbol", "GENE_SYMBOL"]


def find_header_row_gz(path: Path, max_lines: int = 600) -> int | None:
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', errors='ignore') as f:
        for i, line in enumerate(f):
            if i > max_lines:
                break
            cols = line.rstrip('\n').split('\t')
            if 'ProbeName' in cols and any(c in cols for c in ('gProcessedSignal', 'gMeanSignal')):
                return i
    return None


def read_agilent_fe(path: Path) -> pd.DataFrame:
    hdr = find_header_row_gz(path)
    if hdr is None:
        raise RuntimeError(f"Could not locate FEATURES header in {path}")
    df = pd.read_csv(
        path,
        sep='\t',
        compression='gzip' if str(path).endswith('.gz') else None,
        skiprows=hdr,
        dtype=str,
        low_memory=False,
    )
    return df


def extract_expression(df: pd.DataFrame) -> pd.DataFrame:
    int_col = next((c for c in INT_CHOICES if c in df.columns), None)
    probe_col = next((c for c in PROBE_KEYS if c in df.columns), None)
    gene_col = next((c for c in GENE_KEYS if c in df.columns), None)
    if int_col is None or probe_col is None:
        raise ValueError('Missing intensity or probe column in FE file')
    if 'ControlType' in df.columns:
        df = df[df['ControlType'].astype(str) == '0']
    out = df[[probe_col, int_col]].rename(columns={probe_col: 'probe', int_col: 'intensity'})
    if gene_col:
        out['gene'] = df[gene_col]
    return out.reset_index(drop=True)


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    X = df.values
    n_rows, n_cols = X.shape
    sort_idx = np.argsort(X, axis=0, kind='mergesort')
    X_sorted = np.take_along_axis(X, sort_idx, axis=0)
    mean_sorted = X_sorted.mean(axis=1, keepdims=True)
    inv = np.empty_like(sort_idx)
    for j in range(n_cols):
        inv[sort_idx[:, j], j] = np.arange(n_rows)
    X_qn = np.take_along_axis(mean_sorted, inv, axis=0)
    return pd.DataFrame(X_qn, index=df.index, columns=df.columns)


def parse_meta(fn: Path):
    name = fn.stem
    if name.endswith('.txt'):
        name = name[:-4]
    m = re.match(r"(GSM\d+)_([A-Za-z0-9]+)_([RS])_(\d+)$", name)
    if m is None:
        raise ValueError(f"Unexpected filename pattern: {fn.name}")
    gsm, subj, cond, t_idx = m.groups()
    return dict(gsm=gsm, subject=subj, condition=cond, t_idx=int(t_idx), file=str(fn))
