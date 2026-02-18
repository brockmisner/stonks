"""
simulation.py — Backtest / Simulation Runner
Generates realistic synthetic BTC price paths and runs the full engine
Validates the full pipeline without live API keys
"""

import math
import random
import time
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .engine import OracleEngine
from .config import EngineConfig
from .models.signal import MarketSnapshot

logger = logging.getLogger(__name__)


@dataclass
class SimConfig:
    n_games: int = 50
    game_duration_sec: float = 300.0     # 5 minutes
    tick_interval_sec: float = 1.0       # 1 tick per second
    initial_price: float = 45000.0       # BTC starting price
    base_sigma: float = 0.0003           # Per-second log vol (~1.8%/hr)
    jump_prob: float = 0.01              # Prob of jump per tick
    jump_size: float = 0.003            # Jump magnitude
    spread_up: float = 0.025            # UP market spread
    spread_down: float = 0.035          # DOWN market spread
    mid_up_initial: float = 0.50        # Starting UP mid
    seed: Optional[int] = 42


def simulate_btc_path(
    start_price: float,
    n_ticks: int,
    sigma_per_tick: float,
    jump_prob: float = 0.01,
    jump_size: float = 0.003,
    seed: Optional[int] = None,
) -> List[float]:
    """GBM with Poisson jumps — realistic BTC microstructure"""
    rng = random.Random(seed)
    prices = [start_price]
    price = start_price

    for _ in range(n_ticks):
        # Diffusive return
        z = rng.gauss(0, 1)
        r_diff = sigma_per_tick * z

        # Jump component
        r_jump = 0.0
        if rng.random() < jump_prob:
            direction = 1 if rng.random() > 0.5 else -1
            r_jump = direction * rng.uniform(jump_size * 0.5, jump_size * 1.5)

        price = price * math.exp(r_diff + r_jump)
        prices.append(price)

    return prices


def make_orderbook(
    oracle_price: float,
    is_up: bool,
    spread: float,
    p_true: float,
) -> Tuple[float, float]:
    """Generate synthetic orderbook around true probability"""
    # Add small noise to market pricing
    noise = random.gauss(0, 0.01)
    mid = max(0.05, min(0.95, p_true + noise))
    half_spread = spread / 2.0
    bid = max(0.01, mid - half_spread)
    ask = min(0.99, mid + half_spread)
    return bid, ask


class Simulator:
    """
    Full-pipeline simulator for the OracleEngine.
    Runs N games of synthetic BTC price paths and reports results.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        sim_config: Optional[SimConfig] = None,
    ):
        self.engine_cfg = engine_config or EngineConfig()
        self.sim_cfg = sim_config or SimConfig()

        if self.sim_cfg.seed is not None:
            random.seed(self.sim_cfg.seed)

    def run(self) -> dict:
        """Run full simulation and return summary"""
        engine = OracleEngine(self.engine_cfg)
        cfg = self.sim_cfg

        print(f"\n{'='*60}")
        print(f"  POLYMARKET ORACLE ENGINE — SIMULATION")
        print(f"  {cfg.n_games} games × {cfg.game_duration_sec:.0f}s")
        print(f"  Bankroll: ${engine.portfolio.bankroll:.2f}")
        print(f"{'='*60}\n")

        base_price = cfg.initial_price
        n_ticks = int(cfg.game_duration_sec / cfg.tick_interval_sec)

        for game_num in range(1, cfg.n_games + 1):
            # Generate price path
            game_seed = (self.sim_cfg.seed or 0) + game_num
            prices = simulate_btc_path(
                start_price=base_price,
                n_ticks=n_ticks,
                sigma_per_tick=cfg.base_sigma,
                jump_prob=cfg.jump_prob,
                jump_size=cfg.jump_size,
                seed=game_seed,
            )

            strike_price = prices[0]
            settlement_price = prices[-1]
            true_direction = "UP" if settlement_price > strike_price else "DOWN"

            # Game timing
            now = time.time()
            game_start = now
            expiry_time = game_start + cfg.game_duration_sec

            game_id = f"GAME_{game_num:04d}"
            engine.start_game(game_id, expiry_time, game_start)

            # Process ticks
            n_signals = 0
            n_trades = 0
            last_action = None

            for tick_idx, price in enumerate(prices):
                tick_time = game_start + tick_idx * cfg.tick_interval_sec
                T = expiry_time - tick_time

                # True probability (approximate, for orderbook generation)
                log_delta = math.log(price / strike_price) if strike_price > 0 else 0.0
                sigma_approx = cfg.base_sigma * math.sqrt(max(T, 1.0))
                from scipy.stats import norm
                p_true = float(norm.cdf(log_delta / sigma_approx)) if sigma_approx > 0 else 0.5

                # Generate synthetic orderbook
                up_bid, up_ask = make_orderbook(price, True, cfg.spread_up, p_true)
                dn_bid, dn_ask = make_orderbook(price, False, cfg.spread_down, 1.0 - p_true)

                # Add Binance lead (1-3 ticks ahead)
                binance_price = price * math.exp(random.gauss(0, cfg.base_sigma * 0.5))

                snapshot = MarketSnapshot(
                    timestamp=tick_time,
                    oracle_price=price,
                    binance_price=binance_price,
                    up_bid=up_bid,
                    up_ask=up_ask,
                    down_bid=dn_bid,
                    down_ask=dn_ask,
                    oracle_staleness_sec=0.5,
                    market_id=game_id,
                )

                signal = engine.process_tick(snapshot, tick_time)
                n_signals += 1

                if signal.execution.action.value in ("BUY_UP", "BUY_DOWN"):
                    n_trades += 1
                    last_action = signal.execution.action.value

            # Settle game
            settlement = engine.settle_game(settlement_price, expiry_time)

            trade_info = ""
            if n_trades > 0:
                pnl = settlement.get("total_pnl", 0)
                pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                trade_info = f"│ {n_trades} trade(s) {pnl_str}"

            print(
                f"  Game {game_num:>3} │ {true_direction:>4} │ "
                f"strike={strike_price:>9.2f} → settle={settlement_price:>9.2f} "
                f"{trade_info}"
            )

            # Use settlement price as next game's starting price
            base_price = settlement_price

        # End session
        print()
        summary = engine.end_session()

        print(f"\n  FINAL RESULTS")
        print(f"  {'─'*40}")
        print(f"  Bankroll:      ${summary.get('bankroll_final', engine.portfolio.bankroll):.2f}")
        print(f"  Session PnL:   ${summary.get('session_pnl', engine.portfolio.session_pnl):.2f}")
        print(f"  Total Trades:  {summary.get('total_trades', engine.portfolio.total_trades)}")
        win_rate = summary.get("win_rate", engine.portfolio.win_rate)
        if win_rate:
            print(f"  Win Rate:      {win_rate:.1%}")
        print(f"  Drawdown:      {engine.portfolio.drawdown:.1%}")

        return summary
