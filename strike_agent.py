"""
agents/strike_agent.py — Strike locking and game integrity
Per AGENTS.md Section 4
"""

import time
import logging
from typing import Optional

from ..models.signal import GameState

logger = logging.getLogger(__name__)

STRIKE_LOCK_WINDOW = 5.0  # seconds — must lock within ±5s of game start


class StrikeAgent:
    """
    Locks strike within ±5 seconds of game start.
    If not locked in time: skip_this_game = True
    
    Never trade on uncertain strike — preserves statistical validity.
    """

    def __init__(self):
        self._current_game: Optional[GameState] = None

    def start_game(
        self,
        game_id: str,
        expiry_time: float,
        current_time: Optional[float] = None,
    ) -> GameState:
        """Initialize a new game. Strike not yet locked."""
        now = current_time or time.time()
        game = GameState(
            game_id=game_id,
            start_time=now,
            expiry_time=expiry_time,
            active=True,
        )
        self._current_game = game
        logger.info(f"Game started: {game_id} | expiry in {expiry_time - now:.0f}s")
        return game

    def try_lock_strike(
        self,
        oracle_price: float,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        Attempt to lock strike. Returns True if locked successfully.
        Must be called within STRIKE_LOCK_WINDOW seconds of game start.
        """
        if self._current_game is None:
            logger.error("StrikeAgent: No active game")
            return False

        if self._current_game.strike_valid:
            return True  # Already locked

        now = current_time or time.time()
        elapsed = now - self._current_game.start_time

        if elapsed > STRIKE_LOCK_WINDOW:
            logger.warning(
                f"Strike lock timeout: {elapsed:.1f}s > {STRIKE_LOCK_WINDOW}s "
                f"| game={self._current_game.game_id} SKIPPING"
            )
            self._current_game.active = False
            return False

        if oracle_price is None or oracle_price <= 0:
            logger.warning("StrikeAgent: Invalid oracle price, cannot lock")
            return False

        self._current_game.strike = oracle_price
        self._current_game.strike_locked_at = now
        self._current_game.strike_valid = True

        logger.info(
            f"Strike locked: {oracle_price:.2f} at t+{elapsed:.2f}s "
            f"| game={self._current_game.game_id}"
        )
        return True

    def end_game(self, settlement_price: float) -> Optional[GameState]:
        """Close the game and return final state"""
        if self._current_game is None:
            return None

        game = self._current_game
        game.active = False
        game.end_time = time.time()

        if game.strike_valid:
            outcome = "UP" if settlement_price > game.strike else "DOWN"
            logger.info(
                f"Game ended: {game.game_id} | "
                f"strike={game.strike:.2f} settlement={settlement_price:.2f} → {outcome}"
            )

        self._current_game = None
        return game

    @property
    def current_game(self) -> Optional[GameState]:
        return self._current_game

    def time_remaining(self, current_time: Optional[float] = None) -> float:
        """Seconds remaining in current game"""
        if self._current_game is None:
            return 0.0
        now = current_time or time.time()
        return max(0.0, self._current_game.expiry_time - now)

    def is_valid(self) -> bool:
        """Returns True if game is active and strike is locked"""
        if self._current_game is None:
            return False
        return self._current_game.active and self._current_game.strike_valid
