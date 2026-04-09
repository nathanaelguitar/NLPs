"""
Polymarket paper trader — simulates trades against real CLOB prices.

No wallet, no Polygon transactions. Fills are simulated at the current
midpoint price. State persists to a JSON file between sessions.

Usage:
    from prediction_markets import PolymarketClient, PolymarketPaperTrader

    client  = PolymarketClient()
    trader  = PolymarketPaperTrader(starting_balance=1000.0)

    # Look up a market
    markets = client.search_markets("Trump election")
    m = markets[0]
    token_id = m["clobTokenIds"][0]   # index 0 = Yes, 1 = No

    # Simulate a $50 buy on the Yes outcome
    fill = trader.buy(token_id, size_usdc=50.0, label=m["question"] + " — YES")
    print(fill)

    # Check portfolio
    print(trader.summary())
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .polymarket_client import PolymarketClient

_DEFAULT_STATE_FILE = Path.home() / ".polymarket_paper_trades.json"


class PolymarketPaperTrader:
    """
    Paper-trading layer on top of PolymarketClient.

    All "prices" are real CLOB midpoint prices fetched live. Fills are
    simulated at that price — no slippage model, no on-chain calls.

    Parameters
    ----------
    starting_balance : float
        Virtual USDC to start with (default 1000).
    state_file : Path | str | None
        Where to persist portfolio state. Defaults to ~/.polymarket_paper_trades.json.
        Pass None to disable persistence (in-memory only).
    """

    def __init__(
        self,
        starting_balance: float = 1000.0,
        state_file: Path | str | None = _DEFAULT_STATE_FILE,
    ):
        self._client = PolymarketClient()
        self._state_path = Path(state_file) if state_file else None
        self._state = self._load_state(starting_balance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def buy(
        self,
        token_id: str,
        size_usdc: float,
        label: str = "",
    ) -> dict:
        """
        Simulate buying `size_usdc` worth of a token at the current midpoint.

        Returns the fill record dict.
        Raises ValueError if balance is insufficient or no midpoint exists.
        """
        price = self._require_midpoint(token_id)
        balance = self._state["balance_usdc"]
        if size_usdc > balance:
            raise ValueError(
                f"Insufficient balance: have ${balance:.2f}, need ${size_usdc:.2f}"
            )
        shares = size_usdc / price
        self._state["balance_usdc"] -= size_usdc

        pos = self._state["positions"].setdefault(
            token_id,
            {"label": label or token_id, "shares": 0.0, "cost_usdc": 0.0},
        )
        if label and not pos["label"]:
            pos["label"] = label
        pos["shares"] += shares
        pos["cost_usdc"] += size_usdc

        fill = self._record_trade("BUY", token_id, label or pos["label"], shares, price, size_usdc)
        self._save_state()
        return fill

    def sell(
        self,
        token_id: str,
        shares: Optional[float] = None,
        label: str = "",
    ) -> dict:
        """
        Simulate selling `shares` of a token at the current midpoint.

        If `shares` is None, sell the entire position.
        Raises ValueError if the position doesn't exist or shares exceed holdings.
        """
        pos = self._state["positions"].get(token_id)
        if not pos or pos["shares"] <= 0:
            raise ValueError(f"No open position for token {token_id}")

        price = self._require_midpoint(token_id)
        if shares is None:
            shares = pos["shares"]
        if shares > pos["shares"]:
            raise ValueError(
                f"Can't sell {shares:.4f} shares — only hold {pos['shares']:.4f}"
            )

        proceeds = shares * price
        pos["shares"] -= shares
        pos["cost_usdc"] -= (shares / (pos["shares"] + shares)) * pos["cost_usdc"]
        self._state["balance_usdc"] += proceeds

        if pos["shares"] < 1e-9:
            del self._state["positions"][token_id]

        fill = self._record_trade(
            "SELL", token_id, label or (pos.get("label") or token_id), shares, price, proceeds
        )
        self._save_state()
        return fill

    def mark_to_market(self) -> dict:
        """
        Fetch current midpoint prices for all open positions and return
        a dict mapping token_id → {label, shares, cost_usdc, current_value, pnl, pnl_pct}.
        """
        result = {}
        for token_id, pos in self._state["positions"].items():
            price = self._client.get_midpoint(token_id)
            if price is None:
                current_value = None
                pnl = None
                pnl_pct = None
            else:
                current_value = pos["shares"] * price
                pnl = current_value - pos["cost_usdc"]
                pnl_pct = (pnl / pos["cost_usdc"] * 100) if pos["cost_usdc"] else 0.0
            result[token_id] = {
                "label":         pos["label"],
                "shares":        pos["shares"],
                "cost_usdc":     pos["cost_usdc"],
                "current_price": price,
                "current_value": current_value,
                "pnl":           pnl,
                "pnl_pct":       pnl_pct,
            }
        return result

    def summary(self) -> dict:
        """
        High-level portfolio snapshot: cash, positions marked to market,
        total equity, and overall P&L.
        """
        mtm = self.mark_to_market()
        position_value = sum(
            v["current_value"] for v in mtm.values() if v["current_value"] is not None
        )
        total_cost = sum(p["cost_usdc"] for p in self._state["positions"].values())
        equity = self._state["balance_usdc"] + position_value
        starting = self._state["starting_balance"]

        return {
            "starting_balance": starting,
            "cash_usdc":        self._state["balance_usdc"],
            "position_value":   position_value,
            "total_equity":     equity,
            "total_pnl":        equity - starting,
            "total_pnl_pct":    (equity - starting) / starting * 100 if starting else 0.0,
            "open_positions":   mtm,
            "trade_count":      len(self._state["trades"]),
        }

    def trade_history(self) -> list[dict]:
        """Return a copy of all historical fills."""
        return list(self._state["trades"])

    def reset(self, starting_balance: Optional[float] = None) -> None:
        """Wipe all positions and trades, optionally changing the starting balance."""
        bal = starting_balance if starting_balance is not None else self._state["starting_balance"]
        self._state = _fresh_state(bal)
        self._save_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_midpoint(self, token_id: str) -> float:
        price = self._client.get_midpoint(token_id)
        if price is None:
            raise ValueError(f"No midpoint price available for token {token_id}")
        return price

    def _record_trade(
        self,
        side: str,
        token_id: str,
        label: str,
        shares: float,
        price: float,
        usdc: float,
    ) -> dict:
        fill = {
            "ts":       int(time.time()),
            "side":     side,
            "token_id": token_id,
            "label":    label,
            "shares":   round(shares, 6),
            "price":    round(price, 6),
            "usdc":     round(usdc, 4),
        }
        self._state["trades"].append(fill)
        return fill

    def _load_state(self, starting_balance: float) -> dict:
        if self._state_path and self._state_path.exists():
            try:
                return json.loads(self._state_path.read_text())
            except (json.JSONDecodeError, KeyError):
                pass
        return _fresh_state(starting_balance)

    def _save_state(self) -> None:
        if self._state_path:
            self._state_path.write_text(json.dumps(self._state, indent=2))


def _fresh_state(starting_balance: float) -> dict:
    return {
        "starting_balance": starting_balance,
        "balance_usdc":     starting_balance,
        "positions":        {},
        "trades":           [],
    }
