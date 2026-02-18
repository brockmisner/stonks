#!/usr/bin/env python3
"""
main.py — Polymarket Oracle-Aware Quant Engine
Entry point for simulation and live trading modes

Usage:
    python main.py simulate              # Run backtest simulation
    python main.py simulate --games 100  # Custom game count
    python main.py live --market <id>    # Live trading (requires API keys)
    python main.py report                # Print calibration from last session
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet down noisy sub-loggers
logging.getLogger("polymarket_engine.agents.volatility_agent").setLevel(logging.WARNING)
logging.getLogger("polymarket_engine.agents.probability_agent").setLevel(logging.WARNING)
logging.getLogger("polymarket_engine.agents.regime_agent").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("main")

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_simulation(
    n_games: int = 50,
    bankroll: float = 1000.0,
    verbose: bool = False,
    sigma: float = 0.0003,
):
    """Run full simulation"""
    from polymarket_engine.config import EngineConfig
    from polymarket_engine.simulation import Simulator, SimConfig

    if verbose:
        logging.getLogger("polymarket_engine").setLevel(logging.DEBUG)

    engine_cfg = EngineConfig(
        initial_bankroll=bankroll,
        simulation_mode=True,
        log_dir="logs",
    )

    sim_cfg = SimConfig(
        n_games=n_games,
        base_sigma=sigma,
        seed=42,
    )

    sim = Simulator(engine_cfg, sim_cfg)
    summary = sim.run()
    return summary


def run_live(market_id: str, bankroll: float = 1000.0):
    """Run live trading mode"""
    import asyncio
    from polymarket_engine.config import EngineConfig
    from polymarket_engine.engine import OracleEngine
    from polymarket_engine.data_agent import DataAgent, BinanceFeed, PolymarketCLOB
    from polymarket_engine.models.signal import MarketSnapshot

    engine_cfg = EngineConfig(
        initial_bankroll=bankroll,
        simulation_mode=False,
    )

    engine = OracleEngine(engine_cfg)

    async def live_loop():
        binance = BinanceFeed()
        clob = PolymarketCLOB(api_key=os.getenv("POLYMARKET_API_KEY"))
        data_agent = DataAgent(binance_feed=binance, clob_feed=clob)

        logger.info(f"Starting live loop for market: {market_id}")
        logger.warning("Oracle feed not connected — set up Chainlink RPC in data_agent.py")

        # Start game (you'd normally get expiry from the market)
        import time
        expiry = time.time() + 300  # 5 min game
        engine.start_game(market_id, expiry)

        async def on_snapshot(snapshot: MarketSnapshot):
            signal = engine.process_tick(snapshot)
            action = signal.execution.action.value
            if action not in ("HOLD", "SKIP"):
                logger.info(
                    f"ACTION: {action} | EV={signal.execution.chosen_ev:.4f} "
                    f"size=${signal.risk.final_size:.2f}"
                )

        await data_agent.run_live(market_id, on_snapshot, interval_sec=1.0)

    asyncio.run(live_loop())


def print_report(session_id: Optional[str] = None):
    """Print calibration report from saved session"""
    import glob
    log_dir = "logs"

    if session_id:
        path = os.path.join(log_dir, f"calibration_{session_id}.json")
    else:
        # Find most recent
        files = sorted(glob.glob(os.path.join(log_dir, "calibration_*.json")))
        if not files:
            print("No calibration files found in logs/")
            return
        path = files[-1]

    with open(path) as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"  CALIBRATION REPORT — {os.path.basename(path)}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Oracle-Aware Quant Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Simulate
    sim_parser = subparsers.add_parser("simulate", help="Run backtest simulation")
    sim_parser.add_argument("--games", type=int, default=50, help="Number of games")
    sim_parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    sim_parser.add_argument("--sigma", type=float, default=0.0003, help="Per-tick BTC vol")
    sim_parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Live
    live_parser = subparsers.add_parser("live", help="Live trading")
    live_parser.add_argument("--market", type=str, required=True, help="Polymarket market ID")
    live_parser.add_argument("--bankroll", type=float, default=1000.0)

    # Report
    report_parser = subparsers.add_parser("report", help="Print calibration report")
    report_parser.add_argument("--session", type=str, default=None)

    args = parser.parse_args()

    if args.command == "simulate" or args.command is None:
        games = getattr(args, "games", 50)
        bankroll = getattr(args, "bankroll", 1000.0)
        sigma = getattr(args, "sigma", 0.0003)
        verbose = getattr(args, "verbose", False)
        run_simulation(games, bankroll, verbose, sigma)

    elif args.command == "live":
        run_live(args.market, args.bankroll)

    elif args.command == "report":
        print_report(getattr(args, "session", None))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
