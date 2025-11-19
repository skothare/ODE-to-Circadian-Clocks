Leloup–Goldbeter model notes

 - `src/models/leloup_goldbeter.py` implements a 14-state PER/CRY/BMAL1/REV-ERB network inspired by Leloup & Goldbeter, JTB (2004). Expand or tune parameter values as needed to match oscillatory properties.

Mapping to measured transcripts (GSE48113):
- PER (encoded: PER1/2/3) -> state(s) for Per proteins / mRNA
- CRY -> state(s) for Cry
- ARNTL (BMAL1) -> BMAL1
- NR1D1 (REV-ERB) -> Rev-Erb

When building a parameter-estimation objective, map ODE state variables to the measured genes above.
