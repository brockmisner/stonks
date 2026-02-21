import asyncio
import csv
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

import numpy as np
import websockets
from scipy.stats import norm

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, OrderType
from py_clob_client.constants import POLYGON

# =========================
# CONFIG (EDIT ME)
# =========================

# --- Feeds ---
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# Your working Polymarket WS endpoint:
POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# You MUST provide a subscribe payload that yields top-of-book for the active market
# Replace "market_id" and channel fields with your working ones.
POLY_SUBSCRIBE_PAYLOAD = {
    "type": "subscribe",
    "channel": "orderbook",
    "market_id": "REPLACE_ME",
}

# --- Execution (CLOB REST) ---
HOST = "https://clob.polymarket.com"
PRIVATE_KEY = "YOUR_PRIVATE_KEY"
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
API_PASSPHRASE = "YOUR_API_PASSPHRASE"

# Token IDs for UP/DOWN (fill these from gamma/market metadata)
UP_TOKEN_ID = "REPLACE_UP_TOKEN"
DOWN_TOKEN_ID = "REPLACE_DOWN_TOKEN"

# --- Strategy knobs (throughput mode) ---
Z_MIN = 0.65
EDGE_BASE = 0.015
SPREAD_MAX = 0.04
PARITY_LO, PARITY_HI = 0.98, 1.10

WINDOW_HI = 180.0
WINDOW_LO = 60.0

FEE_RATE_PEAK = 0.0044  # conservative fee buffer
SLIPPAGE_TICKS = 0.01  # 1c tick buffer for limit price
SHARES = 10.0

# IOC retry behavior
RETRY_DELAY_S = 0.15
MAX_RETRIES = 1

# Rate limiting (throughput control)
MAX_ORDERS_PER_SEC = 3

# --- Logging ---
LOG_DIR = "./logs"
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")
KPIS_CSV = os.path.join(LOG_DIR, "kpis.csv")


# =========================
# STATE
# =========================

def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class PolyBook:
    ts_ms: int
    up_bid: Optional[float] = None
    up_ask: Optional[float] = None
    down_bid: Optional[float] = None
    down_ask: Optional[float] = None


@dataclass
class MarketState:
    # oracle / settlement anchor (use your source)
    btc_price: float = np.nan
    open_price: float = np.nan
    sec_remaining: float = np.nan

    # Polymarket top of book
    book: PolyBook = field(default_factory=lambda: PolyBook(ts_ms=0))

    # sigma (per-minute log vol) — plug your estimator; here static placeholder
    sigma_1m: float = 0.0009


STATE = MarketState()

# track persistence for same-side signals
PERSIST = {
    "side": None,
    "start_ms": None,
}

# order throttle
ORDER_TIMES: Deque[float] = deque(maxlen=1000)


def allow_order() -> bool:
    """Simple rolling rate limiter."""
    t = time.time()
    ORDER_TIMES.append(t)
    recent = [x for x in ORDER_TIMES if t - x <= 1.0]
    return len(recent) <= MAX_ORDERS_PER_SEC


# =========================
# UTILS
# =========================

def dynamic_persistence(edge: float) -> float:
    if edge >= 0.12:
        return 0.05
    if edge >= 0.08:
        return 0.15
    if edge >= 0.05:
        return 0.35
    return 0.70


def ask_cap_for_edge(edge: float) -> float:
    # tighten if edge small, loosen if edge large
    if edge >= 0.12:
        return 0.80
    if edge >= 0.08:
        return 0.75
    if edge >= 0.05:
        return 0.70
    return 0.0


def cone_p_and_z(btc_price: float, open_price: float, sec_remaining: float, sigma_1m: float):
    tau = sec_remaining / 60.0
    if not np.isfinite(tau) or tau <= 0:
        p = 1.0 if btc_price > open_price else 0.0
        z = np.inf if btc_price > open_price else -np.inf
        return p, z
    vol = sigma_1m * np.sqrt(tau)
    if not np.isfinite(vol) or vol <= 0:
        p = 1.0 if btc_price > open_price else 0.0
        z = np.inf if btc_price > open_price else -np.inf
        return p, z
    z = np.log(btc_price / open_price) / vol
    p = float(norm.cdf(z))
    return p, float(z)


def fee_buffer_per_share(price: float) -> float:
    # conservative; true fee curve is lower at extremes
    return FEE_RATE_PEAK * price


def ensure_logs():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "ts_ms",
                    "side",
                    "token_id",
                    "limit_price",
                    "size",
                    "p_cone",
                    "z",
                    "edge",
                    "ask_at_signal",
                    "bid_at_signal",
                    "sec_remaining",
                    "btc_price",
                    "open_price",
                    "result",
                    "order_id",
                    "exec_ms",
                    "retry_idx",
                ]
            )
    if not os.path.exists(KPIS_CSV):
        with open(KPIS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts_ms", "signals", "orders", "fills", "fill_rate", "avg_edge", "avg_slip"])


# =========================
# CLOB CLIENT + EXECUTOR
# =========================

client = ClobClient(
    host=HOST,
    key=PRIVATE_KEY,
    chain_id=POLYGON,
    signature_type=1,
    creds={"key": API_KEY, "secret": API_SECRET, "passphrase": API_PASSPHRASE},
)

thread_pool = ThreadPoolExecutor(max_workers=4)
execution_queue: asyncio.Queue = asyncio.Queue()


def execute_order_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Runs in a thread. Fires IOC order; returns response + timing."""
    t0 = time.time()
    token_id = payload["token_id"]
    limit_price = payload["price"]
    size = payload["size"]
    retry_idx = payload.get("retry_idx", 0)

    try:
        args = OrderArgs(price=limit_price, size=size, side="BUY", token_id=token_id)
        resp = client.create_and_post_order(args, orderType=OrderType.IOC)
        exec_ms = int((time.time() - t0) * 1000)

        ok = bool(resp) and resp.get("success", False)
        return {
            "ok": ok,
            "resp": resp,
            "exec_ms": exec_ms,
            "retry_idx": retry_idx,
        }
    except Exception as e:
        exec_ms = int((time.time() - t0) * 1000)
        return {"ok": False, "resp": {"error": str(e)}, "exec_ms": exec_ms, "retry_idx": retry_idx}


async def execution_loop():
    ensure_logs()
    while True:
        payload = await execution_queue.get()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(thread_pool, execute_order_sync, payload)
        payload["exec_result"] = result
        await on_execution_result(payload)
        execution_queue.task_done()


# =========================
# MARKET WATCHER (brain)
# =========================

def parse_poly(msg: dict) -> Optional[PolyBook]:
    """Parse a Polymarket WS message into top-of-book where possible."""
    if all(k in msg for k in ["up_bid", "up_ask", "down_bid", "down_ask"]):
        return PolyBook(
            ts_ms=now_ms(),
            up_bid=float(msg["up_bid"]),
            up_ask=float(msg["up_ask"]),
            down_bid=float(msg["down_bid"]),
            down_ask=float(msg["down_ask"]),
        )

    if "bids" in msg and "asks" in msg:
        bids = msg.get("bids") or []
        asks = msg.get("asks") or []
        if bids and asks:
            return PolyBook(ts_ms=now_ms(), up_bid=float(bids[0][0]), up_ask=float(asks[0][0]))
    return None


async def polymarket_book_task():
    while True:
        try:
            async with websockets.connect(POLY_WS, ping_interval=20, ping_timeout=10) as ws:
                await ws.send(json.dumps(POLY_SUBSCRIBE_PAYLOAD))
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    book = parse_poly(msg)
                    if book:
                        STATE.book = book
        except Exception as e:
            print("Polymarket WS error:", repr(e))
            await asyncio.sleep(1)


async def binance_trade_task():
    """Fast BTC reference feed; replace with oracle estimate if desired."""
    while True:
        try:
            async with websockets.connect(BINANCE_WS, ping_interval=20, ping_timeout=10) as ws:
                while True:
                    raw = await ws.recv()
                    msg = json.loads(raw)
                    STATE.btc_price = float(msg["p"])
        except Exception as e:
            print("Binance WS error:", repr(e))
            await asyncio.sleep(1)


async def brain_loop():
    ensure_logs()

    signals = 0
    orders = 0
    fills = 0
    slip_samples = []
    edge_samples = []
    last_kpi_ts = time.time()

    while True:
        await asyncio.sleep(0.01)

        book = STATE.book
        if book.up_bid is None or book.up_ask is None:
            continue
        if not np.isfinite(STATE.btc_price) or not np.isfinite(STATE.open_price) or not np.isfinite(
            STATE.sec_remaining
        ):
            continue

        if not (WINDOW_LO <= STATE.sec_remaining <= WINDOW_HI):
            PERSIST["side"] = None
            PERSIST["start_ms"] = None
            continue

        spread_up = abs(book.up_ask - book.up_bid) if (book.up_ask and book.up_bid) else 1e9
        if spread_up > SPREAD_MAX:
            PERSIST["side"] = None
            PERSIST["start_ms"] = None
            continue

        if book.down_ask is not None and book.up_ask is not None:
            if not (PARITY_LO <= (book.up_ask + book.down_ask) <= PARITY_HI):
                PERSIST["side"] = None
                PERSIST["start_ms"] = None
                continue

        p_cone, z = cone_p_and_z(STATE.btc_price, STATE.open_price, STATE.sec_remaining, STATE.sigma_1m)

        up_edge = p_cone - book.up_ask - fee_buffer_per_share(book.up_ask)
        down_ask = book.down_ask if book.down_ask is not None else 1.0
        down_edge = (1.0 - p_cone) - down_ask - fee_buffer_per_share(down_ask)

        if up_edge >= down_edge:
            side = "UP"
            ask = book.up_ask
            bid = book.up_bid
            edge = float(up_edge)
            token_id = UP_TOKEN_ID
            z_ok = z >= Z_MIN
        else:
            side = "DOWN"
            ask = float(book.down_ask) if book.down_ask is not None else 9e9
            bid = float(book.down_bid) if book.down_bid is not None else 0.0
            edge = float(down_edge)
            token_id = DOWN_TOKEN_ID
            z_ok = z <= -Z_MIN

        if not z_ok or edge < EDGE_BASE:
            PERSIST["side"] = None
            PERSIST["start_ms"] = None
            continue

        cap = ask_cap_for_edge(edge)
        if cap <= 0 or ask > cap:
            PERSIST["side"] = None
            PERSIST["start_ms"] = None
            continue

        wait_s = dynamic_persistence(edge)

        tms = now_ms()
        if PERSIST["side"] != side:
            PERSIST["side"] = side
            PERSIST["start_ms"] = tms

        if (tms - PERSIST["start_ms"]) < int(wait_s * 1000):
            continue

        if not allow_order():
            continue

        limit_price = min(ask + SLIPPAGE_TICKS, cap)

        signals += 1
        orders += 1
        edge_samples.append(edge)

        payload = {
            "detect_ts_ms": tms,
            "token_id": token_id,
            "price": round(limit_price, 3),
            "size": SHARES,
            "side": side,
            "p_cone": p_cone,
            "z": z,
            "edge": edge,
            "ask_at_signal": ask,
            "bid_at_signal": bid,
            "sec_remaining": STATE.sec_remaining,
            "btc_price": STATE.btc_price,
            "open_price": STATE.open_price,
            "retry_idx": 0,
        }
        await execution_queue.put(payload)

        await asyncio.sleep(0.05)

        now = time.time()
        if now - last_kpi_ts >= 10:
            fill_rate = fills / orders if orders else 0.0
            avg_edge = float(np.mean(edge_samples)) if edge_samples else 0.0
            avg_slip = float(np.mean(slip_samples)) if slip_samples else 0.0
            with open(KPIS_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([now_ms(), signals, orders, fills, fill_rate, avg_edge, avg_slip])
            last_kpi_ts = now


async def on_execution_result(payload: Dict[str, Any]):
    """Called after each IOC attempt. If miss and still edge-positive, do one retry."""
    ensure_logs()
    result = payload["exec_result"]
    ok = result["ok"]
    resp = result["resp"]
    exec_ms = result["exec_ms"]
    retry_idx = result["retry_idx"]

    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                now_ms(),
                payload["side"],
                payload["token_id"],
                payload["price"],
                payload["size"],
                payload["p_cone"],
                payload["z"],
                payload["edge"],
                payload["ask_at_signal"],
                payload["bid_at_signal"],
                payload["sec_remaining"],
                payload["btc_price"],
                payload["open_price"],
                "FILLED" if ok else "MISS",
                resp.get("orderID") if isinstance(resp, dict) else "",
                exec_ms,
                retry_idx,
            ]
        )

    if ok or retry_idx >= MAX_RETRIES:
        return

    await asyncio.sleep(RETRY_DELAY_S)

    book = STATE.book
    if book.up_bid is None or book.up_ask is None:
        return
    if not np.isfinite(STATE.btc_price) or not np.isfinite(STATE.open_price) or not np.isfinite(
        STATE.sec_remaining
    ):
        return

    p_cone, z = cone_p_and_z(STATE.btc_price, STATE.open_price, STATE.sec_remaining, STATE.sigma_1m)

    up_edge = p_cone - book.up_ask - fee_buffer_per_share(book.up_ask)
    down_ask = book.down_ask if book.down_ask is not None else 1.0
    down_edge = (1.0 - p_cone) - down_ask - fee_buffer_per_share(down_ask)

    if up_edge >= down_edge:
        side = "UP"
        ask = book.up_ask
        edge = float(up_edge)
        token_id = UP_TOKEN_ID
        z_ok = z >= Z_MIN
    else:
        side = "DOWN"
        ask = float(book.down_ask) if book.down_ask is not None else 9e9
        edge = float(down_edge)
        token_id = DOWN_TOKEN_ID
        z_ok = z <= -Z_MIN

    if not z_ok or edge < EDGE_BASE:
        return

    cap = ask_cap_for_edge(edge)
    if cap <= 0 or ask > cap:
        return

    limit_price = min(ask + 2 * SLIPPAGE_TICKS, cap)

    if not allow_order():
        return

    payload2 = dict(payload)
    payload2.update(
        {
            "detect_ts_ms": now_ms(),
            "token_id": token_id,
            "price": round(limit_price, 3),
            "side": side,
            "p_cone": p_cone,
            "z": z,
            "edge": edge,
            "ask_at_signal": ask,
            "retry_idx": retry_idx + 1,
        }
    )
    await execution_queue.put(payload2)


async def main():
    ensure_logs()
    await asyncio.gather(
        binance_trade_task(),
        polymarket_book_task(),
        execution_loop(),
        brain_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())
