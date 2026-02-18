"""
agents/volatility_agent.py — JumpAwarePhiModel
Computes σ_diff, σ_jump, σ_flip → σ_eff
All calculations in log-return space per MATH.md
"""

import math
import logging
from collections import deque
from typing import Optional

from ..config import VolatilityConfig
from ..models.signal import VolatilityState

logger = logging.getLogger(__name__)


class VolatilityAgent:
    """
    Implements the three-component volatility model:
        σ_eff = sqrt(σ_diff² + σ_jump² + σ_flip²)

    σ_diff: EWMA of log-return variance (microstructure vol)
    σ_jump: Max log-return in window, scaled by 1/√T
    σ_flip: Flip count × penalty × σ_diff
    """

    def __init__(self, config: VolatilityConfig, T_ref: float = 60.0):
        self.cfg = config
        self.T_ref = T_ref  # Reference T for jump scaling

        # EWMA variance state
        self._sigma2_ewma: float = config.sigma_floor ** 2

        # Price history for jump detection
        self._price_history: deque = deque(maxlen=config.jump_window)

        # Sign history for flip detection
        self._sign_history: deque = deque(maxlen=config.flip_window)

        # Last price for log-return calculation
        self._last_price: Optional[float] = None

    def reset(self):
        """Reset state between games"""
        self._sigma2_ewma = self.cfg.sigma_floor ** 2
        self._price_history.clear()
        self._sign_history.clear()
        self._last_price = None

    def update(self, price: float, strike: float, T: float) -> VolatilityState:
        """
        Process one price tick. Returns full VolatilityState.

        Args:
            price: Current oracle/market price
            strike: Locked strike price
            T: Seconds remaining
        """
        state = VolatilityState()

        # --- Log return ---
        if self._last_price is not None and self._last_price > 0 and price > 0:
            log_return = math.log(price / self._last_price)
        else:
            log_return = 0.0

        state.log_return = log_return

        # --- σ_diff: EWMA of log-return variance ---
        alpha = self.cfg.ewma_alpha
        self._sigma2_ewma = (1 - alpha) * self._sigma2_ewma + alpha * (log_return ** 2)
        sigma_diff = math.sqrt(max(self._sigma2_ewma, 0.0))
        state.sigma_diff = sigma_diff

        # --- σ_jump: Max log-return in window, scaled by 1/√T ---
        self._price_history.append(price)
        sigma_jump = 0.0
        if len(self._price_history) >= 2:
            log_returns_window = []
            prices = list(self._price_history)
            for i in range(1, len(prices)):
                if prices[i - 1] > 0 and prices[i] > 0:
                    lr = abs(math.log(prices[i] / prices[i - 1]))
                    log_returns_window.append(lr)
            if log_returns_window:
                J = max(log_returns_window)
                t_scale = math.sqrt(max(T, 1.0))
                sigma_jump = self.cfg.jump_multiplier * J / t_scale

        state.sigma_jump = sigma_jump

        # --- σ_flip: Sign flip count × penalty × σ_diff ---
        delta = price - strike
        sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
        self._sign_history.append(sign)

        flip_count = 0
        signs = list(self._sign_history)
        for i in range(1, len(signs)):
            if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]:
                flip_count += 1

        sigma_flip = flip_count * self.cfg.flip_penalty * sigma_diff
        state.flip_count = flip_count
        state.sigma_flip = sigma_flip

        # Flip rate (flips per tick)
        window_size = len(self._sign_history)
        state.flip_rate = flip_count / window_size if window_size > 0 else 0.0

        # --- σ_eff: Combined effective volatility ---
        sigma_eff_raw = math.sqrt(
            sigma_diff ** 2 +
            sigma_jump ** 2 +
            sigma_flip ** 2
        )

        # Apply floor — never allow sigma to collapse
        sigma_eff = max(sigma_eff_raw, self.cfg.sigma_floor)
        state.sigma_eff = sigma_eff

        # --- Jump intensity: sigma_jump / sigma_eff ---
        if sigma_eff > 0:
            state.jump_intensity = sigma_jump / sigma_eff
        else:
            state.jump_intensity = 0.0

        # Update last price
        self._last_price = price

        logger.debug(
            f"Vol | σ_diff={sigma_diff:.8f} σ_jump={sigma_jump:.8f} "
            f"σ_flip={sigma_flip:.8f} σ_eff={sigma_eff:.8f} "
            f"flips={flip_count} flip_rate={state.flip_rate:.4f}"
        )

        return state

    @property
    def current_sigma_eff(self) -> float:
        return math.sqrt(max(self._sigma2_ewma, 0.0))
