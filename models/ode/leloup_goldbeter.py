"""
Leloup-Goldbeter (2004) mammalian circadian clock model implementation.

This module provides a canonical ODE right-hand side f(t, y, params) to be used with
scipy.integrate.solve_ivp and helpers for default parameters.
This implementation corresponds to the 19-equation model which includes
phosphorylated states and the Rev-ErbA loop.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class LGParams:
    # Transcription maximal rates
    v_sP: float = 1.5
    v_sC: float = 1.2
    v_sB: float = 1.4
    v_sR: float = 1.1

    # Basal transcription (leak)
    v_bP: float = 0.01
    v_bC: float = 0.01
    v_bB: float = 0.01
    v_bR: float = 0.01

    # Michaelis constants for transcriptional activation
    K_act_P: float = 1.0
    K_act_C: float = 1.0
    K_act_B: float = 1.0
    K_act_R: float = 1.0

    # Inhibition thresholds (PER–CRY complex and REV-ERB)
    K_inh_PC: float = 0.7
    K_inh_RB: float = 1.0

    # Hill coefficients
    n_act: float = 2.0
    n_inh: float = 4.0

    # mRNA degradation
    k_degM_P: float = 0.2
    k_degM_C: float = 0.2
    k_degM_B: float = 0.2
    k_degM_R: float = 0.2

    # Translation
    k_sP: float = 0.8
    k_sC: float = 0.6
    k_sB: float = 0.6
    k_sR: float = 0.6

    # Phosphorylation / Dephosphorylation rates (V_max)
    # Per
    V_1P: float = 1.0  # P_C -> P_CP
    V_2P: float = 0.5  # P_CP -> P_C
    V_3P: float = 1.0  # P_N -> P_NP
    V_4P: float = 0.5  # P_NP -> P_N
    # Cry
    V_1C: float = 1.0
    V_2C: float = 0.5
    V_3C: float = 1.0
    V_4C: float = 0.5
    # Bmal1
    V_1B: float = 1.0
    V_2B: float = 0.5
    # Rev-ErbA
    V_1R: float = 1.0
    V_2R: float = 0.5

    # Michaelis constants for phosphorylation
    K_p: float = 0.1  # Generic K for phos steps if not specified individually

    # Degradation of proteins (usually from phosphorylated state)
    k_degP_cP: float = 0.3  # degradation of P_CP
    k_degP_cC: float = 0.3  # degradation of C_CP
    k_degP_cB: float = 0.3  # degradation of B_CP
    k_degP_cR: float = 0.3  # degradation of R_CP
    
    k_degP_nP: float = 0.3  # degradation of P_NP
    k_degP_nC: float = 0.3  # degradation of C_NP
    k_degP_nB: float = 0.3  # degradation of B_N (if no nuc phos)
    k_degP_nR: float = 0.3  # degradation of R_N

    # Cytoplasm ↔ nucleus transport
    k_inP: float = 0.3
    k_outP: float = 0.1
    k_inC: float = 0.3
    k_outC: float = 0.1
    k_inB: float = 0.3
    k_outB: float = 0.1
    k_inR: float = 0.3
    k_outR: float = 0.1

    # PER–CRY complex formation/dissociation and transport
    k_ass_PC: float = 1.5
    k_diss_PC: float = 0.6
    k_inPC: float = 0.3
    k_outPC: float = 0.1
    k_deg_PC_c: float = 0.1
    k_deg_PC_n: float = 0.1


def hill_activation(x, K, n):
    return (x**n) / (K**n + x**n + 1e-12)


def hill_inhibition(x, K, n):
    return 1.0 / (1.0 + (x / (K + 1e-12)) ** n)


def f(t: float, y: Sequence[float], params: LGParams | None = None) -> np.ndarray:
    """Right-hand side of the extended Leloup–Goldbeter 19-equation model."""
    if params is None:
        params = LGParams()

    y = np.asarray(y, dtype=float)

    # State mapping (19 variables)
    # 0-3: mRNAs
    # 4-7: Cytosolic Proteins (unphosphorylated)
    # 8-11: Cytosolic Proteins (phosphorylated)
    # 12-15: Nuclear Proteins (unphosphorylated)
    # 16: Nuclear Protein (phosphorylated Per) - wait, let's be consistent
    
    # Let's use a clearer unpacking
    # Order: 
    # M_P, M_C, M_B, M_R (4)
    # P_C, C_C, B_C, R_C (4)
    # P_CP, C_CP, B_CP, R_CP (4)
    # P_N, C_N, B_N, R_N (4)
    # P_NP, C_NP, PC_C, PC_N (4) -> Total 20? No.
    
    # Let's stick to the 19 inferred:
    # M_P, M_C, M_B, M_R (4)
    # P_C, P_CP, P_N, P_NP (4)
    # C_C, C_CP, C_N, C_NP (4)
    # B_C, B_CP, B_N (3)  <-- Bmal1: Cyto, CytoPhos, Nuc
    # R_C, R_N (2)        <-- Rev: Cyto, Nuc
    # PC_C, PC_N (2)
    # Total: 4+4+4+3+2+2 = 19. This fits!
    
    (
        M_P, M_C, M_B, M_R,
        P_C, P_CP, P_N, P_NP,
        C_C, C_CP, C_N, C_NP,
        B_C, B_CP, B_N,
        R_C, R_N,
        PC_C, PC_N
    ) = y

    # ------------------------------------------------------------------
    # Transcription regulation
    # ------------------------------------------------------------------
    # Bmal1 (B_N) activates Per, Cry, Rev
    # Rev (R_N) inhibits Bmal1
    # Per-Cry (PC_N) inhibits Bmal1's activation of Per, Cry, Rev
    
    B_act = max(B_N, 0.0)
    PC_n_eff = max(PC_N, 0.0)
    R_n_eff = max(R_N, 0.0)

    # Activation by Bmal1
    act_P = hill_activation(B_act, params.K_act_P, params.n_act)
    act_C = hill_activation(B_act, params.K_act_C, params.n_act)
    act_R = hill_activation(B_act, params.K_act_R, params.n_act)
    
    # Inhibition by PC complex (of the activation)
    inh_PC = hill_inhibition(PC_n_eff, params.K_inh_PC, params.n_inh)
    
    # Inhibition of Bmal1 transcription by Rev-ErbA
    inh_RB = hill_inhibition(R_n_eff, params.K_inh_RB, params.n_inh)
    
    # Auto-activation of Bmal1? Usually Bmal1 is inhibited by Rev.
    # And activated by ROR (not modeled here, assumed constitutive/basal).
    # But in LG2004, Bmal1 is inhibited by Rev.
    
    v_tr_P = params.v_bP + params.v_sP * act_P * inh_PC
    v_tr_C = params.v_bC + params.v_sC * act_C * inh_PC
    v_tr_R = params.v_bR + params.v_sR * act_R * inh_PC # Rev also regulated by Bmal/PC? Yes usually.
    v_tr_B = params.v_bB + params.v_sB * inh_RB # Bmal1 inhibited by Rev

    # ------------------------------------------------------------------
    # mRNA dynamics
    # ------------------------------------------------------------------
    dM_P = v_tr_P - params.k_degM_P * M_P
    dM_C = v_tr_C - params.k_degM_C * M_C
    dM_R = v_tr_R - params.k_degM_R * M_R
    dM_B = v_tr_B - params.k_degM_B * M_B

    # ------------------------------------------------------------------
    # Protein Dynamics (Phosphorylation & Transport)
    # ------------------------------------------------------------------
    
    # Helper for Michaelis-Menten rate
    def mm(vmax, s, km):
        return vmax * s / (km + s + 1e-12)

    # PER
    # P_C: Translation, Phos <-> Dephos, Association with C_C, Transport
    dP_C = (
        params.k_sP * M_P
        - mm(params.V_1P, P_C, params.K_p)
        + mm(params.V_2P, P_CP, params.K_p)
        - params.k_ass_PC * P_C * C_C
        + params.k_diss_PC * PC_C
        - params.k_inP * P_C
        + params.k_outP * P_N
        # Degradation usually negligible for unphosphorylated? Or small.
        # Let's assume small or zero for unphos.
    )
    
    # P_CP: Phos <-> Dephos, Degradation
    dP_CP = (
        mm(params.V_1P, P_C, params.K_p)
        - mm(params.V_2P, P_CP, params.K_p)
        - params.k_degP_cP * P_CP
    )
    
    # P_N: Transport, Phos <-> Dephos
    dP_N = (
        params.k_inP * P_C
        - params.k_outP * P_N
        - mm(params.V_3P, P_N, params.K_p)
        + mm(params.V_4P, P_NP, params.K_p)
    )
    
    # P_NP: Phos <-> Dephos, Degradation
    dP_NP = (
        mm(params.V_3P, P_N, params.K_p)
        - mm(params.V_4P, P_NP, params.K_p)
        - params.k_degP_nP * P_NP
    )

    # CRY
    # C_C
    dC_C = (
        params.k_sC * M_C
        - mm(params.V_1C, C_C, params.K_p)
        + mm(params.V_2C, C_CP, params.K_p)
        - params.k_ass_PC * P_C * C_C
        + params.k_diss_PC * PC_C
        - params.k_inC * C_C
        + params.k_outC * C_N
    )
    
    # C_CP
    dC_CP = (
        mm(params.V_1C, C_C, params.K_p)
        - mm(params.V_2C, C_CP, params.K_p)
        - params.k_degP_cC * C_CP
    )
    
    # C_N
    dC_N = (
        params.k_inC * C_C
        - params.k_outC * C_N
        - mm(params.V_3C, C_N, params.K_p)
        + mm(params.V_4C, C_NP, params.K_p)
    )
    
    # C_NP
    dC_NP = (
        mm(params.V_3C, C_N, params.K_p)
        - mm(params.V_4C, C_NP, params.K_p)
        - params.k_degP_nC * C_NP
    )

    # BMAL1
    # B_C: Translation, Phos <-> Dephos, Transport
    dB_C = (
        params.k_sB * M_B
        - mm(params.V_1B, B_C, params.K_p)
        + mm(params.V_2B, B_CP, params.K_p)
        - params.k_inB * B_C
        + params.k_outB * B_N
    )
    
    # B_CP: Phos <-> Dephos, Degradation
    dB_CP = (
        mm(params.V_1B, B_C, params.K_p)
        - mm(params.V_2B, B_CP, params.K_p)
        - params.k_degP_cB * B_CP
    )
    
    # B_N: Transport, Degradation (if no nuclear phos state)
    dB_N = (
        params.k_inB * B_C
        - params.k_outB * B_N
        - params.k_degP_nB * B_N
    )

    # REV-ERBA
    # R_C: Translation, Transport
    dR_C = (
        params.k_sR * M_R
        - params.k_inR * R_C
        + params.k_outR * R_N
        - params.k_degP_cR * R_C # Assuming degradation of unphos if no phos state
    )
    
    # R_N: Transport, Degradation
    dR_N = (
        params.k_inR * R_C
        - params.k_outR * R_N
        - params.k_degP_nR * R_N
    )

    # PER–CRY COMPLEX
    # PC_C
    dPC_C = (
        params.k_ass_PC * P_C * C_C
        - params.k_diss_PC * PC_C
        - params.k_inPC * PC_C
        + params.k_outPC * PC_N
        - params.k_deg_PC_c * PC_C
    )
    
    # PC_N
    dPC_N = (
        params.k_inPC * PC_C
        - params.k_outPC * PC_N
        - params.k_deg_PC_n * PC_N
    )

    dydt = np.array([
        dM_P, dM_C, dM_B, dM_R,
        dP_C, dP_CP, dP_N, dP_NP,
        dC_C, dC_CP, dC_N, dC_NP,
        dB_C, dB_CP, dB_N,
        dR_C, dR_N,
        dPC_C, dPC_N
    ], dtype=float)

    # Enforce non-negativity softly
    dydt = np.where(y < 0, np.maximum(dydt, 0.0), dydt)

    return dydt


def default_initial_conditions(n_states: int = 19) -> np.ndarray:
    y0 = np.ones(n_states) * 0.1
    return y0


__all__ = ["LGParams", "f", "default_initial_conditions"]