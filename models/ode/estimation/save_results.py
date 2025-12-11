"""
Utilities to save parameter estimation results to JSON and CSV.
"""

import json
from pathlib import Path
import pandas as pd


def save_fit_to_json(params: dict, results_dir: str, filename: str = "fit.json"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / filename, "w") as fh:
        json.dump(params, fh, indent=2)


def save_fits_table(fits: list, results_dir: str, filename: str = "fits.csv"):
    df = pd.DataFrame(fits)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / filename, index=False)
