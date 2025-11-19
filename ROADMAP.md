# Project Roadmap

This document outlines the milestones and week-by-week plan for the ODE-to-Circadian-Clocks project.

## Aim 1: Mechanistic Model Implementation
**Goal**: Implement and verify the Leloup–Goldbeter (2004) mammalian circadian clock model.

- [x] **Step 1.1**: Implement canonical ODEs in Python (`src/models/leloup_goldbeter.py`).
    - *Status*: Completed. Upgraded to full 19-equation model including phosphorylated states and Rev-ErbA loop.
- [x] **Step 1.2**: Implement default parameters (`LGParams`).
    - *Status*: Completed.
- [x] **Step 1.3**: Verify oscillatory behavior and positivity (`tests/test_leloup_goldbeter.py`).
    - *Status*: Completed. Tests passing for 19-equation model.

## Aim 2: Data Pipeline & Parameter Estimation
**Goal**: Build a reproducible pipeline for GSE48113 data and estimate model parameters.

- [x] **Step 2.1**: Data Preprocessing (`src/preprocessing/gse48113.py`).
    - *Status*: Implemented.
- [x] **Step 2.2**: Parameter Estimation Scaffolding (`src/estimation/estimation.py`).
    - *Status*: Implemented. Supports local (`least_squares`) and global (`differential_evolution`) optimization.
- [ ] **Step 2.3**: Run Estimation on GSE48113 Data.
    - *Status*: Pending execution (Notebook `02_estimation.ipynb`).

## Aim 3: Identifiability & ML Extensions
**Goal**: Evaluate identifiability and compare mechanistic model with ML approaches.

- [x] **Step 3.1**: Neural ODE Implementation (`src/ml/neural_ode.py`).
    - *Status*: Completed. Basic Neural ODE class implemented using PyTorch.
- [ ] **Step 3.2**: SINDy Implementation (`src/ml/sindy_skeleton.py`).
    - *Status*: Skeleton available. Needs full implementation and testing.
- [ ] **Step 3.3**: Identifiability Analysis (`src/analysis/identifiability.py`).
    - *Status*: Scaffolding available.
- [ ] **Step 3.4**: Compare Neural ODE vs. Mechanistic Model.
    - *Status*: Pending execution (Notebook `05_neural_ode_vs_mechanistic.ipynb`).

## Future Work
- [ ] Integrate SINDy for equation discovery.
- [ ] Perform comprehensive profile likelihood analysis.
- [ ] Finalize comparison figures.
