"""
agents/execution_agent.py — Decision Engine
EV calculation, time cone, stability gates, maker/taker logic
Per AGENTS.md Section 7 + EXECUTION.md
"""

import math
import logging
from typing import Optional

from ..config import ExecutionConfig
from ..models.signal import (
    ProbabilityState, VolatilityState, RegimeState,
    ExecutionDecision, Action, Regime
)

logger = logging.getLogger(__name__)


class ExecutionAgent:
    """
    DecisionEngine implements:
        - Time cone gate (30s < T ≤ 120s)
        - Stability gate (flips > 1 → HOLD)
        - Distance gate (|delta| < 0.5σ√T → HOLD)
        - z gate (|z| < min_z → HOLD)
        - Fee model: fee = p(1-p) * 0.0625
        - EV calculation for UP and DOWN
        - Maker/Taker selection
        - Slippage-adjusted EV test
    """

    def __init__(self, config: ExecutionConfig):
        self.cfg = config

    def compute_fee(self, price: float) -> float:
        """Taker fee model: fee = p(1-p) * r"""
        return price * (1.0 - price) * self.cfg.taker_fee_rate

    def compute_ev_up(self, p_up: float, ask_up: float) -> float:
        """EV for buying UP at ask"""
        fee = self.compute_fee(ask_up)
        return p_up * (1.0 - ask_up) - (1.0 - p_up) * ask_up - fee

    def compute_ev_down(self, p_up: float, ask_down: float) -> float:
        """EV for buying DOWN at ask"""
        p_down = 1.0 - p_up
        fee = self.compute_fee(ask_down)
        return p_down * (1.0 - ask_down) - p_up * ask_down - fee

    def compute_ev_at_price(self, p_win: float, entry: float) -> float:
        """EV at arbitrary entry price"""
        fee = self.compute_fee(entry)
        return p_win * (1.0 - entry) - (1.0 - p_win) * entry - fee

    def decide(
        self,
        T: float,
        oracle_price: float,
        strike: float,
        prob_state: ProbabilityState,
        vol_state: VolatilityState,
        regime_state: RegimeState,
        up_bid: Optional[float],
        up_ask: Optional[float],
        down_bid: Optional[float],
        down_ask: Optional[float],
        oracle_staleness: float = 0.0,
    ) -> ExecutionDecision:
        """
        Full decision pipeline. Returns ExecutionDecision with action and details.
        """
        decision = ExecutionDecision()

        # ─── HARD NO-TRADE CONDITIONS ────────────────────────────────────────

        if oracle_staleness > self.cfg.oracle_max_staleness:
            decision.action = Action.SKIP
            decision.hold_reason = f"Oracle stale: {oracle_staleness:.0f}s > {self.cfg.oracle_max_staleness}s"
            return decision

        if T < self.cfg.min_t_hard:
            decision.action = Action.SKIP
            decision.hold_reason = f"T={T:.1f}s too small (hard min={self.cfg.min_t_hard}s)"
            return decision

        if up_ask is None or down_ask is None:
            decision.action = Action.SKIP
            decision.hold_reason = "Missing bid/ask data"
            return decision

        # ─── TIME CONE GATE ──────────────────────────────────────────────────
        if not (self.cfg.cone_min_t < T <= self.cfg.cone_max_t):
            decision.action = Action.HOLD
            decision.hold_reason = f"Outside time cone: T={T:.1f}s (need {self.cfg.cone_min_t}–{self.cfg.cone_max_t}s)"
            return decision

        # ─── STABILITY GATE ──────────────────────────────────────────────────
        if vol_state.flip_count > self.cfg.max_flips_for_trade:
            decision.action = Action.HOLD
            decision.hold_reason = f"Flip instability: {vol_state.flip_count} flips"
            return decision

        # ─── DISTANCE GATE ───────────────────────────────────────────────────
        sigma_sqrt_T = vol_state.sigma_eff * math.sqrt(max(T, 1.0))
        price_delta = abs(oracle_price - strike)
        min_distance = self.cfg.delta_gate_factor * sigma_sqrt_T

        if price_delta < min_distance:
            decision.action = Action.HOLD
            decision.hold_reason = (
                f"Too close to strike: |Δ|={price_delta:.2f} < {min_distance:.4f}"
            )
            return decision

        # ─── Z GATE ──────────────────────────────────────────────────────────
        z_req = regime_state.z_requirement
        if abs(prob_state.z_adj) < z_req:
            decision.action = Action.HOLD
            decision.hold_reason = (
                f"|z_adj|={abs(prob_state.z_adj):.3f} < z_req={z_req:.2f} "
                f"(regime={regime_state.regime.value})"
            )
            return decision

        # ─── SPREAD VALIDATION ───────────────────────────────────────────────
        spread_up = max(0.0, up_ask - (up_bid or 0.0))
        spread_down = max(0.0, down_ask - (down_bid or 0.0))

        if spread_up < 0 or spread_down < 0:
            decision.action = Action.SKIP
            decision.hold_reason = "Negative spread (data error)"
            return decision

        # ─── EV CALCULATION ──────────────────────────────────────────────────
        p_up = prob_state.p_up_final
        ev_up = self.compute_ev_up(p_up, up_ask)
        ev_down = self.compute_ev_down(p_up, down_ask)
        decision.ev_up = ev_up
        decision.ev_down = ev_down

        dynamic_min_ev = regime_state.dynamic_min_ev

        # ─── DETERMINE BEST SIDE ─────────────────────────────────────────────
        best_side = None
        best_ev = -999.0
        best_ask = 0.0
        best_spread = 0.0

        if ev_up > dynamic_min_ev and ev_up > ev_down:
            best_side = "UP"
            best_ev = ev_up
            best_ask = up_ask
            best_spread = spread_up
        elif ev_down > dynamic_min_ev:
            best_side = "DOWN"
            best_ev = ev_down
            best_ask = down_ask
            best_spread = spread_down

        if best_side is None:
            decision.action = Action.HOLD
            decision.hold_reason = (
                f"EV below threshold: EV_up={ev_up:.4f} EV_down={ev_down:.4f} "
                f"min_ev={dynamic_min_ev:.4f}"
            )
            return decision

        # ─── SPREAD TOXICITY CHECK ───────────────────────────────────────────
        if best_spread > self.cfg.spread_toxic:
            decision.action = Action.SKIP
            decision.hold_reason = f"Toxic spread: {best_spread:.3f} > {self.cfg.spread_toxic}"
            return decision

        # ─── EXECUTION MODE SELECTION ─────────────────────────────────────────
        execution_mode = self._select_execution_mode(T, best_spread, regime_state)

        # ─── SLIPPAGE-ADJUSTED EV TEST (for taker) ───────────────────────────
        p_win = p_up if best_side == "UP" else (1.0 - p_up)
        slip = min(0.02, 0.5 * best_spread)
        ev_after_slip = self.compute_ev_at_price(p_win, best_ask + slip) - self.cfg.slippage_buffer

        if execution_mode == "TAKER" and ev_after_slip < dynamic_min_ev:
            decision.action = Action.HOLD
            decision.hold_reason = (
                f"EV after slippage too low: {ev_after_slip:.4f} < {dynamic_min_ev:.4f}"
            )
            return decision

        # ─── MAKER LIMIT PRICE ───────────────────────────────────────────────
        limit_price = None
        if execution_mode == "MAKER":
            lam = self._maker_lambda(regime_state.regime)
            limit_price = best_ask - lam * best_spread
            limit_price = max(0.01, min(0.99, limit_price))

        # ─── SUCCESS ─────────────────────────────────────────────────────────
        decision.action = Action.BUY_UP if best_side == "UP" else Action.BUY_DOWN
        decision.side = best_side
        decision.chosen_ev = best_ev
        decision.entry_price = limit_price if limit_price else best_ask
        decision.fee = self.compute_fee(best_ask)
        decision.execution_mode = execution_mode
        decision.limit_price = limit_price
        decision.spread = best_spread
        decision.ev_after_slippage = ev_after_slip

        logger.info(
            f"SIGNAL | {decision.action.value} @ {decision.entry_price:.4f} "
            f"EV={best_ev:.4f} ({execution_mode}) "
            f"P(UP)={p_up:.4f} z={prob_state.z_adj:.3f} "
            f"T={T:.0f}s regime={regime_state.regime.value}"
        )

        return decision

    def _select_execution_mode(
        self,
        T: float,
        spread: float,
        regime_state: RegimeState,
    ) -> str:
        """
        Per EXECUTION.md:
        - T < maker_min_T → TAKER
        - spread > spread_max → TAKER
        - ADVERSARIAL → TAKER
        - else → MAKER
        """
        if T < self.cfg.maker_min_t:
            return "TAKER"
        if spread > self.cfg.spread_max_maker:
            return "TAKER"
        if regime_state.regime == Regime.ADVERSARIAL:
            return "TAKER"
        return "MAKER"

    def _maker_lambda(self, regime: Regime) -> float:
        """Lambda for limit placement: L = ask - λ*spread"""
        mapping = {
            Regime.CALM: 0.6,
            Regime.NORMAL: 0.5,
            Regime.HIGH_VOL: 0.4,
            Regime.ADVERSARIAL: 0.2,
        }
        return mapping.get(regime, 0.5)
