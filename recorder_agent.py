"""
agents/recorder_agent.py — Logging, calibration, and performance tracking
Per AGENTS.md Section 10 + CALIBRATION.md
"""

import json
import math
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional, Tuple

from ..models.signal import TradingSignal, PortfolioState

logger = logging.getLogger(__name__)


class RecorderAgent:
    """
    Logs every tick, trade, and outcome.
    Computes calibration metrics after each session:
        - Brier Score
        - Calibration Buckets (predicted vs actual win rate)
        - Profit Factor
        - MAE/MFE analysis
        - EV distribution
    """

    def __init__(self, log_dir: str = "logs", session_id: Optional[str] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.session_id = session_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.trade_log: List[dict] = []
        self.tick_log: List[dict] = []

        # File handles
        self._trade_path = os.path.join(log_dir, f"trades_{self.session_id}.jsonl")
        self._tick_path = os.path.join(log_dir, f"ticks_{self.session_id}.jsonl")
        self._calibration_path = os.path.join(log_dir, f"calibration_{self.session_id}.json")
        self._session_path = os.path.join(log_dir, f"session_{self.session_id}.json")

    def log_tick(self, signal: TradingSignal):
        """Log every tick (pre-decision state)"""
        record = {
            "ts": signal.timestamp,
            "market_id": signal.market_id,
            "T": signal.T,
            "oracle": signal.oracle_price,
            "binance": signal.binance_price,
            "strike": signal.strike,
            "sigma_eff": signal.volatility.sigma_eff,
            "sigma_diff": signal.volatility.sigma_diff,
            "sigma_jump": signal.volatility.sigma_jump,
            "sigma_flip": signal.volatility.sigma_flip,
            "flip_count": signal.volatility.flip_count,
            "flip_rate": signal.volatility.flip_rate,
            "z_raw": signal.probability.z_raw,
            "z_adj": signal.probability.z_adj,
            "p_up": signal.probability.p_up_final,
            "delta": signal.probability.delta,
            "gamma_risk": signal.probability.gamma_risk,
            "regime": signal.regime.regime.value,
            "sigma_ratio": signal.regime.sigma_ratio,
            "dynamic_min_ev": signal.regime.dynamic_min_ev,
            "action": signal.execution.action.value,
            "ev_up": signal.execution.ev_up,
            "ev_down": signal.execution.ev_down,
            "chosen_ev": signal.execution.chosen_ev,
            "entry_price": signal.execution.entry_price,
            "hold_reason": signal.execution.hold_reason,
        }
        self.tick_log.append(record)
        with open(self._tick_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_trade(self, signal: TradingSignal):
        """Log a trade decision (when action is BUY_UP or BUY_DOWN)"""
        record = {
            "ts": signal.timestamp,
            "market_id": signal.market_id,
            "side": signal.execution.side,
            "regime": signal.regime.regime.value,
            "T": signal.T,
            "sigma_eff": signal.volatility.sigma_eff,
            "sigma_ratio": signal.regime.sigma_ratio,
            "flip_count": signal.volatility.flip_count,
            "z_adj": signal.probability.z_adj,
            "p_up": signal.probability.p_up_final,
            "ev_up": signal.execution.ev_up,
            "ev_down": signal.execution.ev_down,
            "chosen_ev": signal.execution.chosen_ev,
            "dynamic_min_ev": signal.regime.dynamic_min_ev,
            "entry_price": signal.execution.entry_price,
            "fee": signal.execution.fee,
            "spread": signal.execution.spread,
            "execution_mode": signal.execution.execution_mode,
            "kelly_fraction": signal.risk.kelly_fraction,
            "final_size": signal.risk.final_size,
            "stability": signal.risk.stability,
            # Post-trade (None until settlement)
            "outcome": signal.outcome,
            "realized_pnl": signal.realized_pnl,
            "mae": signal.mae,
            "mfe": signal.mfe,
            "fill_price": signal.fill_price,
        }
        self.trade_log.append(record)
        with open(self._trade_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def update_trade_outcome(
        self,
        trade_ts: float,
        outcome: int,
        pnl: float,
        mae: float,
        mfe: float,
        fill_price: Optional[float] = None,
    ):
        """Update outcome for a previously logged trade"""
        for record in reversed(self.trade_log):
            if record["ts"] == trade_ts:
                record["outcome"] = outcome
                record["realized_pnl"] = pnl
                record["mae"] = mae
                record["mfe"] = mfe
                record["fill_price"] = fill_price
                break

        # Rewrite the full trade log (simple approach for small logs)
        with open(self._trade_path, "w") as f:
            for record in self.trade_log:
                f.write(json.dumps(record) + "\n")

    def compute_calibration(self) -> dict:
        """
        Compute calibration metrics from completed trades.
        Requires outcome to be filled in.
        """
        completed = [t for t in self.trade_log if t.get("outcome") is not None]
        N = len(completed)

        if N == 0:
            return {"error": "No completed trades", "n": 0}

        # ─── BRIER SCORE ──────────────────────────────────────────────────────
        brier = sum(
            (t["p_up"] - t["outcome"]) ** 2
            for t in completed
        ) / N

        # ─── CALIBRATION BUCKETS ──────────────────────────────────────────────
        # Group by predicted probability of the trade side
        buckets = defaultdict(list)
        for t in completed:
            if t["side"] == "UP":
                predicted = t["p_up"]
                actual = t["outcome"]
            else:
                predicted = 1.0 - t["p_up"]
                actual = t["outcome"]

            bucket_key = round(int(predicted * 10) / 10, 1)
            buckets[bucket_key].append(actual)

        calibration_table = {}
        for key in sorted(buckets):
            vals = buckets[key]
            calibration_table[f"{key:.1f}"] = {
                "predicted": key,
                "actual_win_rate": sum(vals) / len(vals),
                "count": len(vals),
                "gap": abs(key - sum(vals) / len(vals)),
            }

        # ─── PROFIT FACTOR ────────────────────────────────────────────────────
        gross_profit = sum(t["realized_pnl"] for t in completed if (t.get("realized_pnl") or 0) > 0)
        gross_loss = abs(sum(t["realized_pnl"] for t in completed if (t.get("realized_pnl") or 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # ─── MAE / MFE ───────────────────────────────────────────────────────
        maes = [t["mae"] for t in completed if t.get("mae") is not None]
        mfes = [t["mfe"] for t in completed if t.get("mfe") is not None]
        avg_mae = sum(maes) / len(maes) if maes else 0.0
        avg_mfe = sum(mfes) / len(mfes) if mfes else 0.0

        # ─── WIN RATE ─────────────────────────────────────────────────────────
        win_rate = sum(1 for t in completed if t["outcome"] == 1) / N

        # ─── EV REALIZED vs PREDICTED ─────────────────────────────────────────
        avg_ev_predicted = sum(abs(t["chosen_ev"]) for t in completed) / N
        pnls = [t.get("realized_pnl", 0) or 0 for t in completed]
        sizes = [t.get("final_size", 1) or 1 for t in completed]
        avg_realized_return = sum(p / s for p, s in zip(pnls, sizes) if s > 0) / N if N > 0 else 0.0

        metrics = {
            "session_id": self.session_id,
            "n_trades": N,
            "brier_score": round(brier, 5),
            "brier_target": 0.20,
            "brier_ok": brier < 0.20,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "profit_factor_ok": profit_factor > 1.3,
            "avg_mae": round(avg_mae, 4),
            "avg_mfe": round(avg_mfe, 4),
            "mae_mfe_ratio": round(avg_mae / avg_mfe, 3) if avg_mfe > 0 else None,
            "avg_ev_predicted": round(avg_ev_predicted, 5),
            "avg_realized_return": round(avg_realized_return, 5),
            "calibration_table": calibration_table,
        }

        # Write to file
        with open(self._calibration_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def save_session(self, portfolio: PortfolioState):
        """Save final session summary"""
        summary = {
            "session_id": self.session_id,
            "bankroll_final": portfolio.bankroll,
            "session_pnl": portfolio.session_pnl,
            "total_trades": portfolio.total_trades,
            "wins": portfolio.wins,
            "losses": portfolio.losses,
            "win_rate": portfolio.win_rate,
            "drawdown": portfolio.drawdown,
            "calibration": self.compute_calibration(),
        }
        with open(self._session_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Session saved | trades={portfolio.total_trades} "
            f"wins={portfolio.wins} pnl=${portfolio.session_pnl:.2f} "
            f"bankroll=${portfolio.bankroll:.2f}"
        )
        return summary

    def print_calibration_report(self):
        """Pretty-print calibration report to console"""
        metrics = self.compute_calibration()
        if "error" in metrics:
            print(f"  No trades to calibrate: {metrics['error']}")
            return

        print(f"\n{'='*60}")
        print(f"  CALIBRATION REPORT — Session {self.session_id}")
        print(f"{'='*60}")
        print(f"  Trades:         {metrics['n_trades']}")
        print(f"  Win Rate:       {metrics['win_rate']:.1%}")
        print(f"  Brier Score:    {metrics['brier_score']:.5f}  {'✓' if metrics['brier_ok'] else '✗ (target <0.20)'}")
        print(f"  Profit Factor:  {metrics['profit_factor']:.4f}  {'✓' if metrics['profit_factor_ok'] else '✗ (target >1.3)'}")
        print(f"  Avg MAE:        {metrics['avg_mae']:.4f}")
        print(f"  Avg MFE:        {metrics['avg_mfe']:.4f}")
        print(f"  EV Predicted:   {metrics['avg_ev_predicted']:.5f}")
        print(f"  EV Realized:    {metrics['avg_realized_return']:.5f}")

        print(f"\n  Calibration Buckets:")
        print(f"  {'Bucket':>8} {'Predicted':>10} {'Actual':>10} {'Gap':>8} {'Count':>6}")
        print(f"  {'-'*46}")
        for bucket, vals in metrics["calibration_table"].items():
            flag = "⚠" if vals["gap"] > 0.10 else " "
            print(
                f"  {bucket:>8} {vals['predicted']:>10.1%} "
                f"{vals['actual_win_rate']:>10.1%} "
                f"{vals['gap']:>8.1%} {vals['count']:>6} {flag}"
            )
        print(f"{'='*60}\n")
