# Polymarket Oracle-Aware Quant Architecture

---

# 1. Philosophy

This system is not a price predictor.

It is a **probability calibration engine operating against a settlement oracle with friction.**

Core objective:

> Convert live BTC price, volatility structure, and oracle mechanics
> into calibrated settlement probability
> then trade only when EV > friction + regime risk.

Everything else is secondary.

---

# 2. System Architecture Overview

The engine is composed of 7 logical agents (layers):

| Layer | Agent             | Responsibility                           |
| ----- | ----------------- | ---------------------------------------- |
| 1     | Data Agent        | RTDS + Binance + CLOB ingestion          |
| 2     | Strike Agent      | Locks strike and enforces game integrity |
| 3     | Volatility Agent  | Computes σ_diff, σ_jump, σ_flip          |
| 4     | Probability Agent | Computes z and Φ(z)                      |
| 5     | Execution Agent   | Calculates EV and determines action      |
| 6     | Risk Agent        | Kelly sizing + regime filtering          |
| 7     | Recorder Agent    | Logs everything for calibration          |

Each layer must remain modular and testable.

---

# 3. Data Agent

### Sources

* Chainlink Oracle (RTDS)
* Binance (RTDS or direct)
* Polymarket CLOB REST

### Invariants

* Oracle = settlement truth
* Binance = leading indicator
* CLOB ask prices = executable reality

Never mix these roles.

---

# 4. Strike Agent

Strike must be locked within ±5 seconds of game start.

If not:

```
skip_this_game = True
```

Never trade on uncertain strike.

This preserves statistical validity.

---

# 5. Volatility Agent (JumpAwarePhiModel)

This is the mathematical core.

### σ Components

1. σ_diffusion

   * EWMA of log-return variance
   * Base volatility estimate

2. σ_jump

   * Max change in log(price/strike) in window
   * Scaled by 1/√T

3. σ_flip

   * Flip count × penalty × σ_diff
   * Penalizes whip behavior

Final:

[
\sigma_{eff} = \sqrt{σ_{diff}^2 + σ_{jump}^2 + σ_{flip}^2}
]

All calculations must be in **log-return space**.

---

# 6. Probability Agent

Correct z calculation:

[
z = \frac{\ln(S / K)}{σ_{eff} \sqrt{T}}
]

Probability:

[
P(UP) = Φ(z)
]

Never use dollar delta.

Never mix return units and price units.

---

# 7. Execution Agent (DecisionEngine)

### Time Cone

Trade only when:

```
30s < T <= 120s
```

### Stability Gate

```
if flips > 1 → HOLD
```

### Distance Gate

```
if |delta| < 0.5 × σ_eff × √T → HOLD
```

### z Gate

```
if |z| < 1.0 → HOLD
```

### Fee Model

Taker fee:

[
fee = price × (1 − price) × 0.0625
]

EV:

[
EV_{up} = P(UP)(1 − ask) − (1 − P(UP))ask − fee
]

[
EV_{down} = (1 − P(UP))(1 − ask_{down}) − P(UP)ask_{down} − fee
]

Trade only if:

```
EV > min_ev
```

---

# 8. Risk Agent

Fractional Kelly:

[
f = \min(f_{max}, f_{base} × \frac{EV}{EV_{ref}} × stability)
]

Where:

* f_max = 0.05
* EV_ref = 0.02
* stability = max(0.25, 1 − flips / 4)

Position size:

```
size = bankroll × f
```

Never full Kelly.

Never exceed 5%.

---

# 9. Regime Agent (Advanced)

Used for adversarial conditions.

Suppress trading when:

* flip_rate high
* σ_eff extreme
* basis unstable
* oracle staleness high

Optional extension:

```
if σ_eff > 2 × median(σ_eff) → raise min_ev
```

---

# 10. Recorder Agent

Always log:

* timestamp
* oracle price
* binance price
* strike
* σ_eff
* z
* model_p
* EV_up
* EV_down
* chosen action
* entry price
* outcome
* MAE
* MFE

Calibration is mandatory.

---

# 11. Calibration Targets

After each session evaluate:

* Brier score (target < 0.30)
* Calibration bucket gap
* Profit factor (> 1.2 target)
* EV distribution symmetry
* z distribution sanity
* Tail loss size

---

# 12. Design Principles

1. No unit mismatch (log vs dollars)
2. Never trade outside cone
3. Never trade unstable strike
4. Always account for fees
5. Size smaller in unstable regimes
6. Preserve statistical integrity over aggressiveness

---

# 13. Failure Modes To Monitor

* σ_eff = 0 (bad)
* oracle_price = None
* strike_price = None
* basis stuck at 0 permanently
* too many 0.9–1.0 probability buckets
* EV extreme outliers

---

# 14. Future Enhancements

* Volatility regime adaptive min_ev
* Maker routing logic (spread-aware)
* EV variance estimator
* Bayesian z shrinkage
* Cross-game meta-learning
* Dynamic flip penalty

---

# 15. Mission Statement

This is not a trading bot.

This is a **real-time probability calibration engine operating under oracle mechanics and microstructure friction.**

We trade mispricing.
We avoid traps.
We size conservatively.
We measure everything.
