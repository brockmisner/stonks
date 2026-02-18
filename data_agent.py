"""
data_agent.py — Data ingestion for live trading
Connectors for: Chainlink Oracle (RTDS), Binance, Polymarket CLOB REST

Per AGENTS.md Section 3:
    Oracle = settlement truth
    Binance = leading indicator
    CLOB ask prices = executable reality
    
    NEVER mix these roles.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any

from .models.signal import MarketSnapshot

logger = logging.getLogger(__name__)


# ─── BASE INTERFACE ────────────────────────────────────────────────────────────

class PriceFeed(ABC):
    """Base class for all price feeds"""

    @abstractmethod
    async def get_latest(self) -> Optional[float]:
        """Return latest price"""
        pass

    @property
    @abstractmethod
    def staleness_sec(self) -> float:
        """Seconds since last update"""
        pass


class CLOBFeed(ABC):
    """Base class for order book feeds"""

    @abstractmethod
    async def get_orderbook(self, market_id: str) -> Dict[str, Optional[float]]:
        """Return bid/ask for UP and DOWN sides"""
        pass


# ─── CHAINLINK ORACLE CONNECTOR ────────────────────────────────────────────────

class ChainlinkOracleFeed(PriceFeed):
    """
    Chainlink RTDS oracle feed.
    Settlement truth — never use Binance for this role.
    
    Requires: chainlink-data-feeds or direct RPC call
    
    Example endpoint: https://data.chain.link/feeds/ethereum/mainnet/btc-usd
    """

    def __init__(self, rpc_url: Optional[str] = None, feed_address: Optional[str] = None):
        self.rpc_url = rpc_url
        self.feed_address = feed_address
        self._last_price: Optional[float] = None
        self._last_update: float = 0.0

        # BTC/USD Chainlink feed on Ethereum mainnet
        self.BTC_USD_FEED = feed_address or "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88"

    async def get_latest(self) -> Optional[float]:
        """Fetch latest round data from Chainlink"""
        try:
            # Implementation requires web3.py
            # from web3 import Web3
            # w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            # aggregator_abi = [...]  # ABI for latestRoundData()
            # contract = w3.eth.contract(...)
            # _, price, _, updatedAt, _ = contract.functions.latestRoundData().call()
            # self._last_price = price / 1e8  # Chainlink uses 8 decimals
            # self._last_update = time.time()

            # Fallback: return last known price
            return self._last_price

        except Exception as e:
            logger.error(f"Chainlink feed error: {e}")
            return self._last_price

    @property
    def staleness_sec(self) -> float:
        if self._last_update == 0:
            return float("inf")
        return time.time() - self._last_update


# ─── BINANCE CONNECTOR ─────────────────────────────────────────────────────────

class BinanceFeed(PriceFeed):
    """
    Binance WebSocket feed for BTCUSDT.
    Leading indicator only — never use for settlement.
    """

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        self._last_price: Optional[float] = None
        self._last_update: float = 0.0
        self._ws = None

    async def connect(self):
        """Connect to Binance WebSocket"""
        try:
            import websockets
            import json

            url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"
            logger.info(f"Connecting to Binance WS: {url}")

            async with websockets.connect(url) as ws:
                self._ws = ws
                async for message in ws:
                    data = json.loads(message)
                    price = float(data.get("p", 0))
                    if price > 0:
                        self._last_price = price
                        self._last_update = time.time()

        except ImportError:
            logger.warning("websockets not installed. Install: pip install websockets")
        except Exception as e:
            logger.error(f"Binance WS error: {e}")

    async def get_latest(self) -> Optional[float]:
        return self._last_price

    @property
    def staleness_sec(self) -> float:
        if self._last_update == 0:
            return float("inf")
        return time.time() - self._last_update

    async def get_rest(self) -> Optional[float]:
        """REST fallback — single price fetch"""
        try:
            import aiohttp
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    data = await resp.json()
                    price = float(data.get("price", 0))
                    if price > 0:
                        self._last_price = price
                        self._last_update = time.time()
                    return self._last_price
        except Exception as e:
            logger.error(f"Binance REST error: {e}")
            return self._last_price


# ─── POLYMARKET CLOB CONNECTOR ─────────────────────────────────────────────────

class PolymarketCLOB(CLOBFeed):
    """
    Polymarket CLOB REST connector.
    Provides executable bid/ask prices for UP and DOWN sides.
    
    API docs: https://docs.polymarket.com
    """

    BASE_URL = "https://clob.polymarket.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._session = None

    async def get_orderbook(self, market_id: str) -> Dict[str, Optional[float]]:
        """
        Fetch best bid/ask for UP and DOWN token IDs.
        
        Returns:
            {
                "up_bid": float,
                "up_ask": float,
                "down_bid": float,
                "down_ask": float,
            }
        """
        try:
            import aiohttp
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            url = f"{self.BASE_URL}/book?token_id={market_id}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    data = await resp.json()

                    bids = data.get("bids", [])
                    asks = data.get("asks", [])

                    best_bid = float(bids[0]["price"]) if bids else None
                    best_ask = float(asks[0]["price"]) if asks else None

                    return {
                        "up_bid": best_bid,
                        "up_ask": best_ask,
                        "down_bid": (1.0 - best_ask) if best_ask else None,
                        "down_ask": (1.0 - best_bid) if best_bid else None,
                    }

        except Exception as e:
            logger.error(f"Polymarket CLOB error for {market_id}: {e}")
            return {"up_bid": None, "up_ask": None, "down_bid": None, "down_ask": None}

    async def get_markets(self, limit: int = 10) -> list:
        """Fetch available BTC markets"""
        try:
            import aiohttp
            url = f"{self.BASE_URL}/markets?active=true&closed=false&limit={limit}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    markets = data.get("data", [])
                    btc_markets = [
                        m for m in markets
                        if "BTC" in m.get("question", "").upper()
                    ]
                    return btc_markets
        except Exception as e:
            logger.error(f"Polymarket markets fetch error: {e}")
            return []


# ─── DATA AGENT (ORCHESTRATOR) ────────────────────────────────────────────────

class DataAgent:
    """
    Aggregates all data sources into MarketSnapshot objects.
    
    Invariants (per AGENTS.md):
        Oracle = settlement truth
        Binance = leading indicator
        CLOB ask prices = executable reality
    """

    def __init__(
        self,
        oracle_feed: Optional[PriceFeed] = None,
        binance_feed: Optional[PriceFeed] = None,
        clob_feed: Optional[CLOBFeed] = None,
    ):
        self.oracle = oracle_feed
        self.binance = binance_feed
        self.clob = clob_feed

    async def get_snapshot(self, market_id: str) -> MarketSnapshot:
        """Fetch all data sources and return a unified snapshot"""
        # Fetch concurrently
        oracle_task = self.oracle.get_latest() if self.oracle else asyncio.sleep(0)
        binance_task = self.binance.get_latest() if self.binance else asyncio.sleep(0)
        clob_task = self.clob.get_orderbook(market_id) if self.clob else asyncio.sleep(0)

        oracle_price, binance_price, clob_data = await asyncio.gather(
            oracle_task, binance_task, clob_task,
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(oracle_price, Exception):
            logger.error(f"Oracle fetch failed: {oracle_price}")
            oracle_price = None
        if isinstance(binance_price, Exception):
            logger.error(f"Binance fetch failed: {binance_price}")
            binance_price = None
        if isinstance(clob_data, Exception):
            logger.error(f"CLOB fetch failed: {clob_data}")
            clob_data = {}

        staleness = self.oracle.staleness_sec if self.oracle else 0.0

        return MarketSnapshot(
            timestamp=time.time(),
            oracle_price=oracle_price,
            binance_price=binance_price,
            up_bid=clob_data.get("up_bid") if isinstance(clob_data, dict) else None,
            up_ask=clob_data.get("up_ask") if isinstance(clob_data, dict) else None,
            down_bid=clob_data.get("down_bid") if isinstance(clob_data, dict) else None,
            down_ask=clob_data.get("down_ask") if isinstance(clob_data, dict) else None,
            oracle_staleness_sec=staleness,
            market_id=market_id,
        )

    async def run_live(
        self,
        market_id: str,
        on_snapshot: Callable[[MarketSnapshot], None],
        interval_sec: float = 1.0,
        max_ticks: Optional[int] = None,
    ):
        """Run live data loop, calling on_snapshot each tick"""
        tick = 0
        while max_ticks is None or tick < max_ticks:
            try:
                snapshot = await self.get_snapshot(market_id)
                on_snapshot(snapshot)
            except Exception as e:
                logger.error(f"DataAgent tick error: {e}")

            await asyncio.sleep(interval_sec)
            tick += 1
