# ODE-to-Circadian-Clocks
This repository implements the Leloup–Goldbeter mammalian circadian ODE model, data and parameter-estimation workflows for GEO dataset GSE48113, identifiability analyses, and scaffolds for machine learning extensions (Neural ODEs and SINDy). The project is organized around three aims from the proposal:
1. Implement and verify the mechanistic Leloup–Goldbeter model in Python (clean ODE module, default parameter set, and tests validating oscillatory behavior).
2. Build a reproducible pipeline to preprocess GSE48113 and estimate model parameters using local and global optimization.
3. Evaluate identifiability and do an ML-mech comparison (profile likelihoods, sensitivity analyses, Neural ODE and SINDy comparisons).

## Structure

- Data policy
1. Run `notebooks/01_load_GSE48113.ipynb` which downloads and converts GSE48113 to a processed CSV

---

## Running notebooks (local)

Notebooks expect the repository to be importable so they can reuse code in `src/`. There are two ways to make that work locally:

- Start the notebook from the repository root (so `Path.cwd()` is the repo root) or
- Keep the kernel working directory in a subfolder (e.g., `notebooks/`) but ensure the first cell sets the repo root and adds `src` to `sys.path` (we add this in each notebook). Example snippet that notebooks include:

```py
from pathlib import Path
import sys
cwd = Path.cwd()
if (cwd / 'src').exists():
	ROOT = cwd
else:
	ROOT = cwd.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import importlib
importlib.invalidate_caches()
```

If you prefer a global solution, also install this repo editable to your environment:

```powershell
pip install -e .
```

Note: If you add new modules to `src/` after starting the notebook kernel, restart the kernel or call `importlib.invalidate_caches()`.

---

## Roadmap & milestones

See `ROADMAP.md` for the week-by-week plan and milestones. Create GitHub issues from these milestones to track progress.
