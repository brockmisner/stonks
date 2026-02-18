"""
agents/probability_agent.py — Probability calibration engine
Computes z → Φ(z) with shrinkage and mixing per MATH.md + CALIBRATION.md
"""

import math
import logging
from scipy.stats import norm

from ..config import ProbabilityConfig
from ..models.signal import VolatilityState, ProbabilityState

logger = logging.getLogger(__name__)

PHI = norm.cdf
phi_pdf = norm.pdf


class ProbabilityAgent:
    """
    Correct z calculation (per MATH.md):
        z = ln(S / K) / (σ_eff * √T)

    With Bayesian shrinkage (per CALIBRATION.md):
        z_adj = z / (1 + λ|z|)

    Final probability with mixing:
        P_final = α * Φ(z_adj) + (1-α) * 0.5
    """

    def __init__(self, config: ProbabilityConfig):
        self.cfg = config

    def compute(
        self,
        oracle_price: float,
        strike: float,
        T: float,
        vol_state: VolatilityState,
        regime_label: str = "NORMAL",
    ) -> ProbabilityState:
        """
        Compute calibrated settlement probability.

        Args:
            oracle_price: Current oracle price (S)
            strike: Locked strike (K)
            T: Seconds remaining
            vol_state: Output from VolatilityAgent
            regime_label: Current regime for calibration tuning
        """
        state = ProbabilityState()

        if oracle_price <= 0 or strike <= 0 or T <= 0 or vol_state.sigma_eff <= 0:
            logger.warning("ProbabilityAgent: Invalid inputs — returning 0.5")
            state.p_up_final = 0.5
            state.p_down_final = 0.5
            return state

        sigma_eff = vol_state.sigma_eff
        sqrt_T = math.sqrt(T)

        # --- z calculation (log-return space, per MATH.md) ---
        # z = ln(S/K) / (σ_eff * √T)
        log_delta = math.log(oracle_price / strike)
        z_raw = log_delta / (sigma_eff * sqrt_T)
        state.z_raw = z_raw

        # --- Z shrinkage (CALIBRATION.md Fix #1) ---
        # z_adj = z / (1 + λ|z|)  — prevents false 99% certainty
        lam = self.cfg.z_shrink_lambda
        z_adj = z_raw / (1.0 + lam * abs(z_raw))
        state.z_adj = z_adj

        # --- Raw probability ---
        p_raw = float(PHI(z_adj))
        state.p_up_raw = p_raw

        # --- Regime-based mixing alpha (CALIBRATION.md Fix #4) ---
        regime_alpha = self._regime_mixing_alpha(regime_label)

        # --- Final probability with mixing (CALIBRATION.md Fix #3) ---
        # P_final = α*P + (1-α)*0.5  — shrinks toward 50% in uncertain regimes
        p_final = regime_alpha * p_raw + (1.0 - regime_alpha) * 0.5
        # Clamp to [0.01, 0.99]
        p_final = max(0.01, min(0.99, p_final))

        state.p_up_final = p_final
        state.p_down_final = 1.0 - p_final

        # --- Digital option delta ---
        # Δ = φ(z_adj) / (σ_eff * √T)
        pdf_z = float(phi_pdf(z_adj))
        delta = pdf_z / (sigma_eff * sqrt_T) if (sigma_eff * sqrt_T) > 0 else 0.0
        state.delta = delta

        # --- Gamma risk flag: |z_adj| < 1.2 means convexity danger zone ---
        state.gamma_risk = abs(z_adj) < 1.2

        logger.debug(
            f"Prob | S={oracle_price:.2f} K={strike:.2f} T={T:.1f}s "
            f"z_raw={z_raw:.4f} z_adj={z_adj:.4f} "
            f"P(UP)={p_final:.4f} δ={delta:.4f} γ_risk={state.gamma_risk}"
        )

        return state

    def _regime_mixing_alpha(self, regime: str) -> float:
        """Return mixing alpha by regime (less confident in volatile regimes)"""
        mapping = {
            "CALM": 0.95,
            "NORMAL": 0.90,
            "HIGH_VOL": 0.85,
            "ADVERSARIAL": 0.75,
        }
        return mapping.get(regime, self.cfg.mixing_alpha)
