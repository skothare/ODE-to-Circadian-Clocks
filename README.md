# ODE-to-Circadian-Clocks

Modeling human circadian clock dynamics using mechanistic ODEs and data-driven approaches (SINDy, Neural ODEs).

## Overview

This repository implements three modeling paradigms for the human circadian clock:

1. **Mechanistic ODE Model** - Leloup-Goldbeter 19-equation model encoding PER/CRY/BMAL1/REV-ERB feedback loops
2. **SINDy** - Sparse Identification of Nonlinear Dynamics with biological priors
3. **Neural ODEs** - Fully data-driven continuous-time dynamics learning

We use the GSE48113 human blood transcriptome dataset (287 microarray samples, 22 subjects) to fit and compare these approaches.

## Project Structure

```
├── models/                    # Model implementations
│   ├── ode/                   # Mechanistic ODE model
│   │   ├── leloup_goldbeter.py      # 19-equation Leloup-Goldbeter model
│   │   ├── estimation/              # Parameter estimation utilities
│   │   ├── run_02_estimation.py     # Main estimation script
│   │   ├── 03_identifiability_analysis.py
│   │   └── summarize_fit.py
│   ├── neural_ode/            # Neural ODE implementation
│   │   └── neuralODE.ipynb          # Training and evaluation
│   └── sindy/                 # SINDy implementation
│       ├── 04_sindy.ipynb           # SINDy analysis notebook
│       └── sindy_skeleton.py
├── preprocessing/             # Data preprocessing
│   ├── 01_load_GSE48113.ipynb       # Data loading
│   ├── datapeek.ipynb               # Data exploration
│   ├── sindy_data_preproc.ipynb     # SINDy-specific preprocessing
│   └── process_data.py              # Preprocessing utilities
├── scripts/                   # Utility scripts
│   ├── debug_model.py
│   └── extract_metrics.py
├── figures/                   # Generated plots
├── data/                      # Raw and processed data (gitignored)
└── sindy,tex.sty              # LaTeX report
```

## Key Results

| Model | Key Finding |
|-------|-------------|
| Leloup-Goldbeter | Captures ~24h periodicity with biologically consistent phase relationships |
| SINDy | One-step predictions work; free-run simulations fail to sustain oscillations |
| Neural ODE | Fits smooth curves but cannot recover intrinsic periodicity |

## Getting Started

1. **Install dependencies**:

   ```bash
   pip install numpy scipy pandas matplotlib torch torchdiffeq pysindy
   ```

2. **Download data**:
   Run `preprocessing/01_load_GSE48113.ipynb` to download and process GSE48113.

3. **Run estimation**:

   ```bash
   cd models/ode
   python run_02_estimation.py
   ```

## Authors

- Achyudhan Kutuva (University of Pittsburgh, Carnegie Mellon University)
- Arth Banka (Carnegie Mellon University)
- Riti Bhatia (Carnegie Mellon University)
- Sanchitha Kuthethoor (Carnegie Mellon University)
- Sumeet Kothare (Carnegie Mellon University)

## References

- Leloup & Goldbeter (2004) - Mammalian circadian clock model
- Archer et al. (2014) - GSE48113 dataset
