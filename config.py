"""
config.py — Central configuration for Polymarket Oracle-Aware Engine
All constants derived from AGENTS.md, MATH.md, RISK.md, REGIME.md
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VolatilityConfig:
    # EWMA alpha for diffusion vol
    ewma_alpha: float = 0.1
    # Floor to prevent sigma collapse → z explosion
    sigma_floor: float = 0.00008
    # Jump detection window (ticks)
    jump_window: int = 20
    # Jump scaling multiplier
    jump_multiplier: float = 1.5
    # Flip penalty factor
    flip_penalty: float = 0.5
    # Flip window (ticks)
    flip_window: int = 20


@dataclass
class ProbabilityConfig:
    # Z shrinkage factor (Bayesian dampening) λ
    z_shrink_lambda: float = 0.10
    # Probability mixing: P_final = alpha*P + (1-alpha)*0.5
    mixing_alpha: float = 0.90
    # Min z to trade
    min_z: float = 1.0


@dataclass
class ExecutionConfig:
    # Trade cone: only trade in [min_T, max_T] seconds
    cone_min_t: float = 30.0
    cone_max_t: float = 120.0
    # Taker fee rate
    taker_fee_rate: float = 0.0625
    # Min EV to trade (base)
    min_ev_base: float = 0.02
    # Max flips before HOLD
    max_flips_for_trade: int = 1
    # Min delta distance gate (fraction of sigma*sqrt(T))
    delta_gate_factor: float = 0.5
    # Maker minimum T
    maker_min_t: float = 25.0
    # Max spread to use maker
    spread_max_maker: float = 0.03
    # Toxic spread threshold
    spread_toxic: float = 0.10
    # Strong EV threshold for taker even with small T
    strong_ev_threshold: float = 0.06
    # Slippage buffer for taker
    slippage_buffer: float = 0.005
    # Spread haircut gamma
    spread_haircut_gamma: float = 8.0
    # Oracle max staleness (seconds)
    oracle_max_staleness: float = 60.0
    # Min T to trade at all
    min_t_hard: float = 5.0


@dataclass
class RiskConfig:
    # Kelly parameters
    f_max: float = 0.05
    f_base: float = 0.02
    ev_ref: float = 0.02
    # Hard cap per trade
    hard_cap_fraction: float = 0.03
    # Session stop loss
    session_stop_loss: float = 0.03
    # Drawdown reduce active allocation
    drawdown_reduce_threshold: float = 0.10
    # Active allocation (vs 20% reserve)
    active_allocation: float = 0.80
    # Consecutive loss guard
    consecutive_loss_limit: int = 3
    # MAE threshold for large loss flag
    mae_large_loss_threshold: float = 0.20
    # Max total exposure
    max_total_exposure: float = 0.08


@dataclass
class RegimeConfig:
    # Sigma ratio thresholds
    calm_sigma_ratio: float = 0.7
    high_vol_sigma_ratio: float = 1.3
    # Flip rate thresholds
    normal_flip_rate: float = 0.08
    adversarial_flip_rate: float = 0.10
    # Jump intensity threshold
    jump_intensity_high: float = 0.5
    # EV adjustment factors
    ev_alpha: float = 0.5   # sigma_ratio weight
    ev_beta: float = 2.0    # flip_rate weight
    # Kelly multipliers per regime
    kelly_multiplier: dict = field(default_factory=lambda: {
        "CALM": 1.0,
        "NORMAL": 1.0,
        "HIGH_VOL": 0.7,
        "ADVERSARIAL": 0.5,
    })
    # Z requirements per regime
    z_requirement: dict = field(default_factory=lambda: {
        "CALM": 1.0,
        "NORMAL": 1.0,
        "HIGH_VOL": 1.2,
        "ADVERSARIAL": 1.5,
    })
    # Regime persistence (ticks before switching)
    persistence_ticks: int = 3
    # Reference sigma (long-run median, updated dynamically)
    sigma_ref_initial: float = 0.000350
    # Volatility expansion threshold
    vol_expansion_pct: float = 0.50


@dataclass
class CalibrationConfig:
    brier_target: float = 0.20
    bucket_gap_target: float = 0.10
    min_trades_for_calibration: int = 20


@dataclass
class EngineConfig:
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    probability: ProbabilityConfig = field(default_factory=ProbabilityConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Starting bankroll
    initial_bankroll: float = 1000.0
    # Log directory
    log_dir: str = "logs"
    # Data directory
    data_dir: str = "data"
    # Simulation mode (no live execution)
    simulation_mode: bool = True


# Regime labels
REGIME_CALM = "CALM"
REGIME_NORMAL = "NORMAL"
REGIME_HIGH_VOL = "HIGH_VOL"
REGIME_ADVERSARIAL = "ADVERSARIAL"

# Action labels
ACTION_BUY_UP = "BUY_UP"
ACTION_BUY_DOWN = "BUY_DOWN"
ACTION_HOLD = "HOLD"
ACTION_SKIP = "SKIP"
