"""
engine.py — Main Orchestrator
Polymarket Oracle-Aware Quant Engine

Coordinates all 7 agents in the correct pipeline order:
    Data → Strike → Volatility → Probability → Regime → Execution → Risk → Record
"""

import logging
import time
from typing import Optional

from .config import EngineConfig
from .models.signal import (
    MarketSnapshot, TradingSignal, PortfolioState, GameState, Action
)
from .agents.strike_agent import StrikeAgent
from .agents.volatility_agent import VolatilityAgent
from .agents.probability_agent import ProbabilityAgent
from .agents.regime_agent import RegimeAgent
from .agents.execution_agent import ExecutionAgent
from .agents.risk_agent import RiskAgent
from .agents.recorder_agent import RecorderAgent

logger = logging.getLogger(__name__)


class OracleEngine:
    """
    Core probability calibration engine.

    Architecture (7 agents, per AGENTS.md):
        1. Data Agent      → MarketSnapshot ingestion (handled externally)
        2. Strike Agent    → Locks strike, enforces game integrity
        3. Volatility Agent → σ_diff + σ_jump + σ_flip → σ_eff
        4. Probability Agent → z + Φ(z) → calibrated P(UP)
        5. Regime Agent    → Regime classification + dynamic thresholds
        6. Execution Agent → EV gates + maker/taker logic → Action
        7. Risk Agent      → Kelly sizing → final position size
        8. Recorder Agent  → Full logging + calibration

    Mission:
        Convert live BTC price, volatility structure, and oracle mechanics
        into calibrated settlement probability,
        then trade only when EV > friction + regime risk.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.cfg = config or EngineConfig()

        # Initialize portfolio
        self.portfolio = PortfolioState(
            bankroll=self.cfg.initial_bankroll,
            peak_bankroll=self.cfg.initial_bankroll,
        )

        # Initialize all agents
        self.strike_agent = StrikeAgent()
        self.volatility_agent = VolatilityAgent(self.cfg.volatility)
        self.probability_agent = ProbabilityAgent(self.cfg.probability)
        self.regime_agent = RegimeAgent(self.cfg.regime)
        self.execution_agent = ExecutionAgent(self.cfg.execution)
        self.risk_agent = RiskAgent(self.cfg.risk, self.cfg.execution)
        self.recorder = RecorderAgent(log_dir=self.cfg.log_dir)

        # Last signal for post-trade updates
        self._pending_trade: Optional[TradingSignal] = None

        logger.info(
            f"OracleEngine initialized | bankroll=${self.portfolio.bankroll:.2f} "
            f"simulation={self.cfg.simulation_mode}"
        )

    def start_game(
        self,
        game_id: str,
        expiry_time: float,
        current_time: Optional[float] = None,
    ) -> GameState:
        """Start a new 5-minute game"""
        self.volatility_agent.reset()
        self.regime_agent.reset()
        game = self.strike_agent.start_game(game_id, expiry_time, current_time)
        logger.info(f"Game started: {game_id}")
        return game

    def process_tick(
        self,
        snapshot: MarketSnapshot,
        current_time: Optional[float] = None,
    ) -> TradingSignal:
        """
        Process one market tick through the full agent pipeline.
        Returns a TradingSignal with action and all intermediate states.

        Args:
            snapshot: Raw market data (oracle, binance, orderbook)
            current_time: Unix timestamp (defaults to time.time())
        """
        now = current_time or time.time()
        signal = TradingSignal(
            timestamp=now,
            market_id=snapshot.market_id,
        )

        # ─── STRIKE LOCK ──────────────────────────────────────────────────────
        if not self.strike_agent.is_valid():
            if snapshot.oracle_price:
                locked = self.strike_agent.try_lock_strike(snapshot.oracle_price, now)
                if not locked:
                    signal.execution.action = Action.SKIP
                    signal.execution.hold_reason = "Strike not locked"
                    self.recorder.log_tick(signal)
                    return signal
            else:
                signal.execution.action = Action.SKIP
                signal.execution.hold_reason = "No oracle price for strike"
                self.recorder.log_tick(signal)
                return signal

        game = self.strike_agent.current_game
        if game is None:
            signal.execution.action = Action.SKIP
            signal.execution.hold_reason = "No active game"
            return signal

        strike = game.strike
        T = self.strike_agent.time_remaining(now)

        signal.strike = strike
        signal.T = T
        signal.oracle_price = snapshot.oracle_price
        signal.binance_price = snapshot.binance_price

        # ─── DATA VALIDATION ──────────────────────────────────────────────────
        if snapshot.oracle_price is None:
            signal.execution.action = Action.SKIP
            signal.execution.hold_reason = "Oracle price is None"
            self.recorder.log_tick(signal)
            return signal

        # ─── VOLATILITY AGENT ─────────────────────────────────────────────────
        vol_state = self.volatility_agent.update(
            price=snapshot.oracle_price,
            strike=strike,
            T=T,
        )
        signal.volatility = vol_state

        # ─── REGIME AGENT ─────────────────────────────────────────────────────
        regime_state = self.regime_agent.update(vol_state)
        signal.regime = regime_state

        # ─── PROBABILITY AGENT ────────────────────────────────────────────────
        prob_state = self.probability_agent.compute(
            oracle_price=snapshot.oracle_price,
            strike=strike,
            T=max(T, 1.0),
            vol_state=vol_state,
            regime_label=regime_state.regime.value,
        )
        signal.probability = prob_state

        # ─── EXECUTION AGENT ──────────────────────────────────────────────────
        exec_decision = self.execution_agent.decide(
            T=T,
            oracle_price=snapshot.oracle_price,
            strike=strike,
            prob_state=prob_state,
            vol_state=vol_state,
            regime_state=regime_state,
            up_bid=snapshot.up_bid,
            up_ask=snapshot.up_ask,
            down_bid=snapshot.down_bid,
            down_ask=snapshot.down_ask,
            oracle_staleness=snapshot.oracle_staleness_sec,
        )
        signal.execution = exec_decision

        # ─── RISK AGENT ───────────────────────────────────────────────────────
        if exec_decision.action in (Action.BUY_UP, Action.BUY_DOWN):
            risk_decision = self.risk_agent.compute_size(
                portfolio=self.portfolio,
                exec_decision=exec_decision,
                prob_state=prob_state,
                vol_state=vol_state,
                regime_state=regime_state,
                sigma_ref=self.regime_agent.sigma_ref,
            )
            signal.risk = risk_decision

            # If risk doesn't approve, downgrade to HOLD
            if not risk_decision.approved:
                signal.execution.action = Action.HOLD
                signal.execution.hold_reason = f"Risk rejected: {risk_decision.rejection_reason}"

        # ─── RECORD ───────────────────────────────────────────────────────────
        self.recorder.log_tick(signal)

        if signal.execution.action in (Action.BUY_UP, Action.BUY_DOWN):
            self.recorder.log_trade(signal)
            self._pending_trade = signal
            logger.info(
                f"TRADE | {signal.execution.action.value} "
                f"size=${signal.risk.final_size:.2f} "
                f"@ {signal.execution.entry_price:.4f} "
                f"EV={signal.execution.chosen_ev:.4f}"
            )

        return signal

    def settle_game(
        self,
        settlement_price: float,
        current_time: Optional[float] = None,
    ) -> dict:
        """
        Settle the game and update portfolio.
        Returns settlement summary.
        """
        game = self.strike_agent.end_game(settlement_price)
        if game is None:
            return {"error": "No game to settle"}

        settled_trades = 0
        total_pnl = 0.0
        results = []

        # Find all pending signals and compute outcomes
        for record in self.recorder.trade_log:
            if record.get("outcome") is not None:
                continue  # Already settled

            side = record.get("side", "")
            strike = game.strike
            size = record.get("final_size", 0) or 0
            entry = record.get("entry_price", 0.5)
            ts = record.get("ts", 0)

            if not side or not strike or size <= 0:
                continue

            # Determine outcome
            if side == "UP":
                won = settlement_price > strike
            else:
                won = settlement_price <= strike

            outcome = 1 if won else 0
            win_payout = 1.0 - entry
            loss_cost = entry

            # PnL in dollars
            pnl = size * win_payout if won else -size * loss_cost

            # Approximate MAE/MFE (in simulation, simplified)
            mae = abs(loss_cost * size) if not won else size * 0.05
            mfe = abs(win_payout * size) if won else size * 0.05

            # Update record
            self.recorder.update_trade_outcome(ts, outcome, pnl, mae, mfe, settlement_price)

            # Update portfolio
            self.portfolio = self.risk_agent.update_portfolio_post_trade(
                self.portfolio, pnl, won, mae
            )

            total_pnl += pnl
            settled_trades += 1
            results.append({
                "side": side,
                "won": won,
                "pnl": round(pnl, 4),
                "size": round(size, 2),
            })

            logger.info(
                f"SETTLE | {side} {'WIN' if won else 'LOSS'} "
                f"pnl=${pnl:.2f} bankroll=${self.portfolio.bankroll:.2f}"
            )

        return {
            "game_id": game.game_id,
            "strike": game.strike,
            "settlement": settlement_price,
            "direction": "UP" if settlement_price > (game.strike or 0) else "DOWN",
            "trades_settled": settled_trades,
            "total_pnl": round(total_pnl, 4),
            "bankroll": round(self.portfolio.bankroll, 2),
            "results": results,
        }

    def end_session(self) -> dict:
        """End trading session, compute and print full calibration"""
        summary = self.recorder.save_session(self.portfolio)
        self.recorder.print_calibration_report()

        logger.info(
            f"Session complete | trades={self.portfolio.total_trades} "
            f"win_rate={self.portfolio.win_rate:.1%} "
            f"pnl=${self.portfolio.session_pnl:.2f} "
            f"bankroll=${self.portfolio.bankroll:.2f}"
        )
        return summary

    @property
    def status(self) -> dict:
        """Quick engine status snapshot"""
        return {
            "bankroll": round(self.portfolio.bankroll, 2),
            "session_pnl": round(self.portfolio.session_pnl, 2),
            "trades": self.portfolio.total_trades,
            "win_rate": round(self.portfolio.win_rate, 4),
            "drawdown": round(self.portfolio.drawdown, 4),
            "regime": self.regime_agent.current_regime.value,
            "sigma_ref": self.regime_agent.sigma_ref,
            "strike_valid": self.strike_agent.is_valid(),
        }
