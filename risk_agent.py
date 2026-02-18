"""
agents/risk_agent.py — Risk Management & Position Sizing
Implements fractional Kelly, tail caps, regime scaling, drawdown control
Per RISK.md + AGENTS.md Section 8
"""

import math
import logging
from typing import Optional

from ..config import RiskConfig, ExecutionConfig
from ..models.signal import (
    ExecutionDecision, ProbabilityState, VolatilityState,
    RegimeState, RiskDecision, PortfolioState, Regime, Action
)

logger = logging.getLogger(__name__)


class RiskAgent:
    """
    Position sizing and risk management:

    Fractional Kelly:
        f = min(f_max, f_base * (EV/EV_ref) * stability)

    Stability:
        stability = max(0.25, 1 - flips/4)

    Then apply:
        - Regime Kelly multiplier
        - Spread haircut
        - Tail risk cap (if σ_eff > 2*median → halve)
        - Hard cap: max 3% bankroll per trade
        - Total exposure cap: max 8% bankroll
    """

    def __init__(self, risk_config: RiskConfig, exec_config: ExecutionConfig):
        self.rcfg = risk_config
        self.ecfg = exec_config

    def compute_size(
        self,
        portfolio: PortfolioState,
        exec_decision: ExecutionDecision,
        prob_state: ProbabilityState,
        vol_state: VolatilityState,
        regime_state: RegimeState,
        sigma_ref: float,
    ) -> RiskDecision:
        """
        Compute final position size in dollars.
        Returns RiskDecision with all intermediate steps.
        """
        decision = RiskDecision()

        # Must have a trade signal
        if exec_decision.action not in (Action.BUY_UP, Action.BUY_DOWN):
            decision.approved = False
            decision.rejection_reason = "No trade signal"
            return decision

        # ─── SESSION STOP LOSS ────────────────────────────────────────────────
        if portfolio.session_pnl < -(self.rcfg.session_stop_loss * portfolio.bankroll):
            decision.approved = False
            decision.rejection_reason = (
                f"Session stop loss triggered: PnL={portfolio.session_pnl:.2f}"
            )
            logger.warning("RISK: Session stop loss — halting trading")
            return decision

        # ─── DRAWDOWN ALLOCATION REDUCTION ───────────────────────────────────
        if portfolio.drawdown > self.rcfg.drawdown_reduce_threshold:
            effective_br = portfolio.bankroll * 0.60
        else:
            effective_br = portfolio.bankroll * portfolio.active_allocation

        # ─── STABILITY ───────────────────────────────────────────────────────
        flips = vol_state.flip_count
        stability = max(0.25, 1.0 - flips / 4.0)
        decision.stability = stability

        # ─── FRACTIONAL KELLY ─────────────────────────────────────────────────
        ev = exec_decision.chosen_ev
        f_raw = self.rcfg.f_base * (ev / self.rcfg.ev_ref) * stability
        f_raw = min(self.rcfg.f_max, f_raw)
        decision.kelly_fraction = f_raw

        # ─── VARIANCE-ADJUSTED KELLY (optional upgrade) ───────────────────────
        # f_var = EV / Var where Var = p(1-p)
        p = prob_state.p_up_final if exec_decision.side == "UP" else prob_state.p_down_final
        variance = p * (1.0 - p)
        if variance > 0:
            f_variance = ev / variance
            # Use the more conservative of the two
            f_raw = min(f_raw, f_variance * 0.25)  # 25% of variance Kelly

        # ─── BASE POSITION SIZE ───────────────────────────────────────────────
        base_size = effective_br * f_raw
        decision.position_size = base_size

        # ─── KELLY REDUCTION IF CONSECUTIVE LOSSES ───────────────────────────
        if portfolio.kelly_reduction_active:
            f_max_reduced = self.rcfg.f_max * 0.5
            base_size = min(base_size, effective_br * f_max_reduced)
            logger.debug(f"Kelly reduction active: size capped at {base_size:.2f}")

        # ─── REGIME SIZE MULTIPLIER ───────────────────────────────────────────
        regime_size_mult = {
            Regime.CALM: 1.0,
            Regime.NORMAL: 1.0,
            Regime.HIGH_VOL: 0.7,
            Regime.ADVERSARIAL: 0.5,
        }.get(regime_state.regime, 1.0)

        size_after_regime = base_size * regime_size_mult
        decision.size_after_regime = size_after_regime

        # ─── TAIL RISK: Volatility expansion → halve size ─────────────────────
        if sigma_ref > 0 and vol_state.sigma_eff > 2.0 * sigma_ref:
            size_after_regime *= 0.5
            logger.debug(
                f"Tail risk cap: σ_eff={vol_state.sigma_eff:.8f} > "
                f"2*σ_ref={2*sigma_ref:.8f} → size halved"
            )

        # ─── SPREAD HAIRCUT ───────────────────────────────────────────────────
        # size = size * (1 - 8 * spread)
        spread = exec_decision.spread
        spread_haircut = max(0.0, 1.0 - self.ecfg.spread_haircut_gamma * spread)
        size_after_spread = size_after_regime * spread_haircut
        decision.size_after_spread_haircut = size_after_spread

        # ─── HARD CAPS ────────────────────────────────────────────────────────
        # Per-trade hard cap: max 3% bankroll
        hard_cap = portfolio.bankroll * self.rcfg.hard_cap_fraction
        final_size = min(size_after_spread, hard_cap)

        # Total exposure cap: max 8% bankroll (across overlapping signals)
        max_exposure = portfolio.bankroll * self.rcfg.max_total_exposure
        final_size = min(final_size, max_exposure)

        # Floor: don't trade with less than $1
        if final_size < 1.0:
            decision.approved = False
            decision.rejection_reason = f"Position too small: ${final_size:.4f}"
            decision.final_size = 0.0
            return decision

        decision.final_size = final_size
        decision.approved = True

        logger.info(
            f"RISK | size=${final_size:.2f} kelly={f_raw:.4f} "
            f"stability={stability:.3f} regime_mult={regime_size_mult:.2f} "
            f"spread_haircut={spread_haircut:.3f} hard_cap=${hard_cap:.2f}"
        )

        return decision

    def update_portfolio_post_trade(
        self,
        portfolio: PortfolioState,
        pnl: float,
        won: bool,
        mae: float = 0.0,
    ) -> PortfolioState:
        """Update portfolio state after trade settles"""
        portfolio.bankroll += pnl
        portfolio.session_pnl += pnl
        portfolio.total_trades += 1

        if won:
            portfolio.wins += 1
            portfolio.consecutive_losses = 0
            portfolio.kelly_reduction_active = False
        else:
            portfolio.losses += 1
            portfolio.consecutive_losses += 1

        # Update peak bankroll
        if portfolio.bankroll > portfolio.peak_bankroll:
            portfolio.peak_bankroll = portfolio.bankroll

        # Drawdown check → reduce active allocation
        if portfolio.drawdown > self.rcfg.drawdown_reduce_threshold:
            portfolio.active_allocation = 0.60
            logger.warning(
                f"Drawdown {portfolio.drawdown:.1%} > threshold → active alloc 60%"
            )
        else:
            portfolio.active_allocation = self.rcfg.active_allocation

        # Consecutive loss guard
        if portfolio.consecutive_losses >= self.rcfg.consecutive_loss_limit:
            portfolio.kelly_reduction_active = True
            portfolio.kelly_reduction_trades_remaining = 5
            logger.warning(
                f"{portfolio.consecutive_losses} consecutive losses → Kelly halved for 5 trades"
            )

        if portfolio.kelly_reduction_trades_remaining > 0:
            portfolio.kelly_reduction_trades_remaining -= 1
            if portfolio.kelly_reduction_trades_remaining == 0:
                portfolio.kelly_reduction_active = False

        return portfolio
