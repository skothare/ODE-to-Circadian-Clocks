import json
from src.estimation.save_results import save_fit_to_json, save_fits_table
from pathlib import Path
import tempfile


def test_save_json_and_csv():
    with tempfile.TemporaryDirectory() as td:
        d = {"params": {"a": 1.0}, "loss": 0.2}
        save_fit_to_json(d, td, filename="testfit.json")
        p = Path(td) / "testfit.json"
        assert p.exists()
        with open(p) as fh:
            dd = json.load(fh)
        assert dd["params"]["a"] == 1.0

        fits = [{"id": 1, "loss": 0.2}, {"id": 2, "loss": 0.15}]
        save_fits_table(fits, td, filename="fits.csv")
        assert (Path(td) / "fits.csv").exists()
