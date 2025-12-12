# ODE-to-Circadian-Clocks

Modeling human circadian clock dynamics using mechanistic ODEs and data-driven approaches (SINDy, Neural ODEs).

## Overview

This repository implements three modeling paradigms for the human circadian clock:

1. **Mechanistic ODE Model** - Leloup-Goldbeter 19-equation model encoding PER/CRY/BMAL1/REV-ERB feedback loops
2. **SINDy** - Sparse Identification of Nonlinear Dynamics with biological priors
3. **Neural ODEs** - Fully data-driven continuous-time dynamics learning

We use the GSE48113 human blood transcriptome dataset (287 microarray samples, 22 subjects) to fit and compare these approaches.

## Key Results

| Model | Key Finding |
|-------|-------------|
| Leloup-Goldbeter | Captures ~24h periodicity with biologically consistent phase relationships |
| SINDy | One-step predictions work; free-run simulations fail to sustain oscillations |
| Neural ODE | Fits smooth curves but cannot recover intrinsic periodicity |

## Authors

- Achyudhan Kutuva (University of Pittsburgh, Carnegie Mellon University)
- Arth Banka (Carnegie Mellon University)
- Riti Bhatia (Carnegie Mellon University)
- Sanchitha Kuthethoor (Carnegie Mellon University)
- Sumeet Kothare (Carnegie Mellon University)

## References

- Leloup & Goldbeter (2004) - Mammalian circadian clock model
- Archer et al. (2014) - [GSE48113 dataset] (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48113)
