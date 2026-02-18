"""
agents/regime_agent.py — Volatility Regime Detection & Classification
Implements the 4-regime state machine from REGIME.md
"""

import logging
from collections import deque
from typing import Optional

from ..config import RegimeConfig, REGIME_CALM, REGIME_NORMAL, REGIME_HIGH_VOL, REGIME_ADVERSARIAL
from ..models.signal import VolatilityState, RegimeState, Regime

logger = logging.getLogger(__name__)


class RegimeAgent:
    """
    4-regime state machine per REGIME.md:

        CALM        → σ_ratio < 0.7,  flip_rate < 0.03
        NORMAL      → 0.7 ≤ σ_ratio ≤ 1.3, flip_rate < 0.08
        HIGH_VOL    → σ_ratio > 1.3  OR jump_intensity > 0.5
        ADVERSARIAL → flip_rate > 0.10 OR >3 flips in 20s

    Includes:
        - Persistence filter (don't switch < 3 consecutive ticks)
        - Dynamic EV threshold adjustment
        - Kelly multiplier scaling
        - Sigma reference (long-run rolling median)
    """

    def __init__(self, config: RegimeConfig):
        self.cfg = config

        # Current confirmed regime
        self._current_regime: Regime = Regime.NORMAL

        # Candidate regime (must persist N ticks to confirm)
        self._candidate_regime: Optional[Regime] = None
        self._candidate_ticks: int = 0

        # Rolling sigma_eff for reference calculation
        self._sigma_history: deque = deque(maxlen=500)
        self._sigma_ref: float = config.sigma_ref_initial

        # Previous sigma_eff for expansion detection
        self._prev_sigma_eff: Optional[float] = None

    def reset(self):
        """Reset between games (keep long-run sigma reference)"""
        self._current_regime = Regime.NORMAL
        self._candidate_regime = None
        self._candidate_ticks = 0
        self._prev_sigma_eff = None

    def update(self, vol_state: VolatilityState) -> RegimeState:
        """
        Process one tick of volatility state and output regime classification.
        """
        sigma_eff = vol_state.sigma_eff
        flip_rate = vol_state.flip_rate
        jump_intensity = vol_state.jump_intensity

        # Update rolling sigma reference (long-run median)
        self._sigma_history.append(sigma_eff)
        if len(self._sigma_history) >= 10:
            sorted_sigs = sorted(self._sigma_history)
            self._sigma_ref = sorted_sigs[len(sorted_sigs) // 2]  # median
        sigma_ref = max(self._sigma_ref, 1e-10)

        sigma_ratio = sigma_eff / sigma_ref

        # --- Classify candidate regime ---
        candidate = self._classify(sigma_ratio, flip_rate, jump_intensity)

        # --- Volatility expansion check (override) ---
        if self._prev_sigma_eff is not None and self._prev_sigma_eff > 0:
            expansion = (sigma_eff - self._prev_sigma_eff) / self._prev_sigma_eff
            if expansion > self.cfg.vol_expansion_pct:
                candidate = Regime.HIGH_VOL
                logger.debug(f"Vol expansion detected: +{expansion:.1%} → HIGH_VOL override")

        self._prev_sigma_eff = sigma_eff

        # --- Persistence filter (switch only after N consecutive ticks) ---
        if candidate != self._current_regime:
            if candidate == self._candidate_regime:
                self._candidate_ticks += 1
            else:
                self._candidate_regime = candidate
                self._candidate_ticks = 1

            if self._candidate_ticks >= self.cfg.persistence_ticks:
                old = self._current_regime
                self._current_regime = candidate
                self._candidate_regime = None
                self._candidate_ticks = 0
                logger.info(f"Regime change: {old.value} → {self._current_regime.value} "
                            f"(σ_ratio={sigma_ratio:.3f} flip_rate={flip_rate:.4f})")
        else:
            self._candidate_regime = None
            self._candidate_ticks = 0

        regime = self._current_regime

        # --- Dynamic EV threshold ---
        # min_ev = base * (1 + α*σ_ratio + β*flip_rate)
        dynamic_min_ev = 0.02 * (
            1.0 +
            self.cfg.ev_alpha * sigma_ratio +
            self.cfg.ev_beta * flip_rate
        )
        # Adversarial: double it
        if regime == Regime.ADVERSARIAL:
            dynamic_min_ev *= 2.0

        # --- Kelly multiplier ---
        kelly_mult = self.cfg.kelly_multiplier.get(regime.value, 1.0)

        # --- Z requirement ---
        z_req = self.cfg.z_requirement.get(regime.value, 1.0)

        # --- Execution mode preference ---
        exec_mode = "TAKER" if regime == Regime.ADVERSARIAL else "MAKER"

        state = RegimeState(
            regime=regime,
            sigma_ratio=sigma_ratio,
            dynamic_min_ev=dynamic_min_ev,
            kelly_multiplier=kelly_mult,
            z_requirement=z_req,
            execution_mode_preferred=exec_mode,
        )

        logger.debug(
            f"Regime | {regime.value} σ_ratio={sigma_ratio:.3f} "
            f"flip_rate={flip_rate:.4f} dyn_min_ev={dynamic_min_ev:.4f} "
            f"kelly_mult={kelly_mult:.2f} z_req={z_req:.2f}"
        )

        return state

    def _classify(
        self,
        sigma_ratio: float,
        flip_rate: float,
        jump_intensity: float,
    ) -> Regime:
        """Pure classification without persistence filter"""
        cfg = self.cfg

        # Adversarial: flip_rate overrides everything
        if flip_rate > cfg.adversarial_flip_rate:
            return Regime.ADVERSARIAL

        # High vol: extreme sigma or jump-dominated
        if sigma_ratio > cfg.high_vol_sigma_ratio or jump_intensity > cfg.jump_intensity_high:
            return Regime.HIGH_VOL

        # Calm: quiet diffusion
        if sigma_ratio < cfg.calm_sigma_ratio and flip_rate < 0.03:
            return Regime.CALM

        # Default
        return Regime.NORMAL

    @property
    def current_regime(self) -> Regime:
        return self._current_regime

    @property
    def sigma_ref(self) -> float:
        return self._sigma_ref
