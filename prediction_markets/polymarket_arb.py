"""
Polymarket Intra-Market Arbitrage Bot
======================================
Strategy
--------
A binary Polymarket market resolves to exactly $1.00 for one outcome (Yes or
No) and $0.00 for the other.  If you hold *both* tokens you are guaranteed to
collect $1.00.  Polymarket charges a 2 % fee on winnings, so the guaranteed
net payout is $0.98.

Arbitrage opportunity fires when:
    best_ask(YES) + best_ask(NO) < PAYOUT_AFTER_FEE - gas_cost_per_leg * 2

The bot streams the live CLOB order book via WebSocket, evaluates this
inequality after every update, and — when an opportunity is confirmed —
simultaneously submits market-fill (FOK) orders for both legs.

Required .env variables (add to prediction_markets/.env)
---------------------------------------------------------
  POLY_PRIVATE_KEY    Hex private key for your Polygon wallet, no 0x prefix
  POLY_API_KEY        Polymarket CLOB API key   (from clob.polymarket.com)
  POLY_API_SECRET     Polymarket CLOB API secret
  POLY_PASSPHRASE     Polymarket CLOB API passphrase
  CONDITION_ID        Target market condition_id  (from Gamma API)
  YES_TOKEN_ID        CLOB asset_id of the YES token
  NO_TOKEN_ID         CLOB asset_id of the NO token
  DRY_RUN             "true" to simulate (default), "false" to trade live
  MAX_POSITION_USDC   Maximum USDC per arb trade (default: 100)
  MIN_NET_PROFIT_PCT  Minimum net profit % to trigger trade (default: 0.5)

Install dependencies
--------------------
    pip install py-clob-client websockets python-dotenv

Usage
-----
    # dry-run (default)
    python -m prediction_markets.polymarket_arb

    # live trading
    DRY_RUN=false python -m prediction_markets.polymarket_arb
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import websockets
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment / configuration
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

CLOB_WS_URL  = "wss://clob.polymarket.com/ws"
CLOB_REST_URL = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137

# Polymarket charges 2 % on winning payouts
WINNING_FEE_RATE = Decimal("0.02")
PAYOUT_AFTER_FEE = Decimal("1.00") - WINNING_FEE_RATE  # $0.98

# Estimated gas cost per single on-chain order fill (USDC).
# Polygon gas is cheap; 0.02 is very conservative.
GAS_COST_PER_LEG_USDC = Decimal("0.02")
TOTAL_GAS_COST = GAS_COST_PER_LEG_USDC * 2  # both legs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "polymarket_arb.log"),
    ],
)
log = logging.getLogger("poly.arb")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All runtime parameters sourced from environment variables."""

    private_key: str          = field(default_factory=lambda: _require_env("POLY_PRIVATE_KEY"))
    api_key: str              = field(default_factory=lambda: _require_env("POLY_API_KEY"))
    api_secret: str           = field(default_factory=lambda: _require_env("POLY_API_SECRET"))
    passphrase: str           = field(default_factory=lambda: _require_env("POLY_PASSPHRASE"))
    condition_id: str         = field(default_factory=lambda: _require_env("CONDITION_ID"))
    yes_token_id: str         = field(default_factory=lambda: _require_env("YES_TOKEN_ID"))
    no_token_id: str          = field(default_factory=lambda: _require_env("NO_TOKEN_ID"))

    dry_run: bool             = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() != "false")
    max_position_usdc: Decimal = field(
        default_factory=lambda: Decimal(os.getenv("MAX_POSITION_USDC", "100"))
    )
    min_net_profit_pct: Decimal = field(
        default_factory=lambda: Decimal(os.getenv("MIN_NET_PROFIT_PCT", "0.5")) / 100
    )

    # WebSocket reconnection
    ws_reconnect_delay_s: float = 3.0
    ws_max_reconnect_attempts: int = 10

    # REST polling fallback interval (seconds) – used if WS is unavailable
    poll_interval_s: float = 2.0


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set.  "
            f"Add it to {_ENV_PATH}"
        )
    return val


@dataclass
class PriceLevel:
    price: Decimal
    size: Decimal


@dataclass
class OrderBook:
    """Single-token order book with incremental update support."""

    token_id: str
    bids: Dict[Decimal, Decimal] = field(default_factory=dict)  # price -> size
    asks: Dict[Decimal, Decimal] = field(default_factory=dict)  # price -> size
    last_updated: float = 0.0

    def apply_snapshot(self, buys: List[dict], sells: List[dict]) -> None:
        self.bids = {Decimal(lvl["price"]): Decimal(lvl["size"]) for lvl in buys}
        self.asks = {Decimal(lvl["price"]): Decimal(lvl["size"]) for lvl in sells}
        self.last_updated = time.time()

    def apply_changes(self, changes: List[dict]) -> None:
        """
        Apply incremental price_change deltas.
        A size of "0" means remove the level.
        """
        for chg in changes:
            price = Decimal(chg["price"])
            size  = Decimal(chg["size"])
            side  = chg["side"].upper()
            book  = self.bids if side == "BUY" else self.asks
            if size == 0:
                book.pop(price, None)
            else:
                book[price] = size
        self.last_updated = time.time()

    def best_ask(self) -> Optional[PriceLevel]:
        """Return the lowest ask level, or None if the asks side is empty."""
        if not self.asks:
            return None
        price = min(self.asks)
        return PriceLevel(price=price, size=self.asks[price])

    def best_bid(self) -> Optional[PriceLevel]:
        if not self.bids:
            return None
        price = max(self.bids)
        return PriceLevel(price=price, size=self.bids[price])

    def available_at_or_below(self, max_price: Decimal) -> Decimal:
        """Total ask liquidity available at or below max_price (USDC worth of shares)."""
        return sum(
            price * size
            for price, size in self.asks.items()
            if price <= max_price
        )


@dataclass
class ArbitrageSignal:
    yes_ask: Decimal
    no_ask: Decimal
    combined_cost: Decimal
    payout: Decimal
    gas_cost: Decimal
    net_profit: Decimal
    net_profit_pct: Decimal
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return (
            f"ARB SIGNAL | YES ask={self.yes_ask:.4f}  NO ask={self.no_ask:.4f}  "
            f"combined={self.combined_cost:.4f}  payout={self.payout:.4f}  "
            f"gas={self.gas_cost:.4f}  net_profit={self.net_profit:.4f} "
            f"({self.net_profit_pct*100:.3f}%)"
        )


@dataclass
class TradeResult:
    signal: ArbitrageSignal
    yes_order_id: Optional[str]
    no_order_id: Optional[str]
    yes_fill_price: Optional[Decimal]
    no_fill_price: Optional[Decimal]
    size_usdc: Decimal
    success: bool
    dry_run: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Risk Manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Controls position sizing and enforces drawdown limits.

    Parameters
    ----------
    max_position_usdc : Decimal
        Hard cap on USDC per individual arb trade.
    min_net_profit_pct : Decimal
        Trades below this net profit fraction are rejected (e.g. 0.005 = 0.5 %).
    max_daily_loss_usdc : Decimal
        Stop all trading for the day once realised losses exceed this.
    max_concurrent_trades : int
        Maximum number of arb legs that may be open simultaneously.
    """

    def __init__(
        self,
        max_position_usdc: Decimal = Decimal("100"),
        min_net_profit_pct: Decimal = Decimal("0.005"),
        max_daily_loss_usdc: Decimal = Decimal("50"),
        max_concurrent_trades: int = 3,
    ) -> None:
        self.max_position_usdc     = max_position_usdc
        self.min_net_profit_pct    = min_net_profit_pct
        self.max_daily_loss_usdc   = max_daily_loss_usdc
        self.max_concurrent_trades = max_concurrent_trades

        self._daily_pnl: Decimal          = Decimal("0")
        self._open_trade_count: int       = 0
        self._trade_history: Deque[dict]  = deque(maxlen=200)
        self._total_pnl: Decimal          = Decimal("0")

    # ------------------------------------------------------------------
    # Pre-trade checks
    # ------------------------------------------------------------------

    def approve_signal(self, signal: ArbitrageSignal) -> Tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        All checks must pass; first failure short-circuits.
        """
        if signal.net_profit_pct < self.min_net_profit_pct:
            return False, (
                f"Net profit {signal.net_profit_pct*100:.3f}% < "
                f"minimum {self.min_net_profit_pct*100:.3f}%"
            )

        if self._daily_pnl <= -self.max_daily_loss_usdc:
            return False, (
                f"Daily loss limit reached: ${self._daily_pnl:.2f} "
                f"(limit ${self.max_daily_loss_usdc:.2f})"
            )

        if self._open_trade_count >= self.max_concurrent_trades:
            return False, (
                f"Max concurrent trades ({self.max_concurrent_trades}) reached"
            )

        return True, "OK"

    def compute_position_size(
        self,
        signal: ArbitrageSignal,
        yes_book: OrderBook,
        no_book: OrderBook,
    ) -> Decimal:
        """
        Return how many USDC to deploy per leg, respecting:
          1. max_position_usdc cap
          2. Available liquidity at the signal's ask prices (no slippage beyond signal)
        """
        yes_liq  = yes_book.available_at_or_below(signal.yes_ask)
        no_liq   = no_book.available_at_or_below(signal.no_ask)
        liq_cap  = min(yes_liq, no_liq)         # bottleneck leg
        size     = min(self.max_position_usdc, liq_cap)
        return max(size, Decimal("0"))           # never negative

    # ------------------------------------------------------------------
    # Post-trade accounting
    # ------------------------------------------------------------------

    def open_trade(self) -> None:
        self._open_trade_count += 1

    def close_trade(self, result: TradeResult) -> None:
        self._open_trade_count = max(0, self._open_trade_count - 1)
        if result.success and result.yes_fill_price and result.no_fill_price:
            cost   = (result.yes_fill_price + result.no_fill_price) * result.size_usdc
            pnl    = (PAYOUT_AFTER_FEE - (result.yes_fill_price + result.no_fill_price)) * result.size_usdc
            self._daily_pnl  += pnl
            self._total_pnl  += pnl
            self._trade_history.append({
                "ts": result.timestamp,
                "pnl_usdc": float(pnl),
                "cost_usdc": float(cost),
                "dry_run": result.dry_run,
            })

    def reset_daily_pnl(self) -> None:
        """Call at UTC midnight to reset the daily loss guard."""
        self._daily_pnl = Decimal("0")

    def stats(self) -> dict:
        wins  = sum(1 for t in self._trade_history if t["pnl_usdc"] > 0)
        total = len(self._trade_history)
        return {
            "total_trades": total,
            "win_rate_pct": round(100 * wins / total, 2) if total else 0,
            "daily_pnl_usdc": float(self._daily_pnl),
            "total_pnl_usdc": float(self._total_pnl),
            "open_trades": self._open_trade_count,
        }


# ---------------------------------------------------------------------------
# Trading Engine
# ---------------------------------------------------------------------------

class TradingEngine:
    """
    Wraps py-clob-client for async order execution.

    All blocking CLOB calls are dispatched via asyncio.to_thread() so the
    event loop is never stalled.  Dry-run mode logs orders but never submits
    them to the network.
    """

    def __init__(self, config: Config) -> None:
        self.config  = config
        self.dry_run = config.dry_run
        self._client = self._build_client(config)
        self._log    = logging.getLogger("poly.arb.engine")

    @staticmethod
    def _build_client(config: Config):
        """
        Construct a py-clob-client ClobClient.
        Import is deferred so the module can be imported even if py-clob-client
        is not installed (e.g. for unit tests in dry-run-only contexts).
        """
        try:
            from py_clob_client.client import ClobClient  # type: ignore
            from py_clob_client.clob_types import ApiCreds  # type: ignore
            creds = ApiCreds(
                api_key=config.api_key,
                api_secret=config.api_secret,
                api_passphrase=config.passphrase,
            )
            return ClobClient(
                host=CLOB_REST_URL,
                key=config.private_key,
                chain_id=POLYGON_CHAIN_ID,
                creds=creds,
            )
        except ImportError:
            if not config.dry_run:
                raise RuntimeError(
                    "py-clob-client is not installed.  "
                    "Run: pip install py-clob-client"
                )
            return None  # dry-run doesn't need the real client

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    async def place_market_order(
        self,
        token_id: str,
        size_usdc: Decimal,
        label: str = "",
    ) -> dict:
        """
        Submit a Fill-Or-Kill market buy order and return the response dict.
        In dry-run mode, returns a synthetic fill without touching the network.
        """
        if self.dry_run:
            self._log.info(
                "[DRY RUN] MARKET BUY  token=%s  size_usdc=%.4f  label=%s",
                token_id[:8] + "…",
                size_usdc,
                label,
            )
            return {"status": "filled", "order_id": f"dry_{token_id[:8]}", "price": None}

        from py_clob_client.clob_types import MarketOrderArgs  # type: ignore
        from py_clob_client.constants import BUY  # type: ignore  – noqa

        # py-clob-client amount is in USDC for market orders
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=float(size_usdc),
        )

        def _submit():
            order = self._client.create_market_order(order_args)
            # FOK = Fill-Or-Kill: ensures atomic fill, no partial open orders
            from py_clob_client.client import OrderType  # type: ignore
            return self._client.post_order(order, OrderType.FOK)

        response = await asyncio.to_thread(_submit)
        self._log.info("Order response: %s", response)
        return response

    async def execute_arbitrage(
        self,
        signal: ArbitrageSignal,
        yes_token_id: str,
        no_token_id: str,
        size_usdc: Decimal,
    ) -> TradeResult:
        """
        Submit both legs of the arbitrage concurrently.

        Both FOK orders are issued in parallel; if either fails the result
        is marked unsuccessful so the caller can risk-manage accordingly.
        """
        self._log.info(
            "Executing arb | size_per_leg=%.4f USDC | %s",
            size_usdc,
            signal,
        )

        yes_task = asyncio.create_task(
            self.place_market_order(yes_token_id, size_usdc, label="YES")
        )
        no_task = asyncio.create_task(
            self.place_market_order(no_token_id, size_usdc, label="NO")
        )

        yes_resp, no_resp = await asyncio.gather(yes_task, no_task, return_exceptions=True)

        # Parse results
        yes_ok    = isinstance(yes_resp, dict) and yes_resp.get("status") == "filled"
        no_ok     = isinstance(no_resp,  dict) and no_resp.get("status")  == "filled"
        success   = yes_ok and no_ok

        error_msg = None
        if not yes_ok:
            error_msg = f"YES leg failed: {yes_resp}"
        if not no_ok:
            err_suffix = f" | NO leg failed: {no_resp}"
            error_msg  = (error_msg or "") + err_suffix

        if error_msg:
            self._log.error("Arb execution failed: %s", error_msg)
        else:
            self._log.info(
                "Arb executed successfully | net_profit_pct=%.3f%%",
                float(signal.net_profit_pct) * 100,
            )

        yes_fill = Decimal(str(yes_resp.get("price") or signal.yes_ask)) if yes_ok else None
        no_fill  = Decimal(str(no_resp.get("price")  or signal.no_ask))  if no_ok  else None

        return TradeResult(
            signal=signal,
            yes_order_id=yes_resp.get("order_id") if yes_ok else None,
            no_order_id=no_resp.get("order_id")   if no_ok  else None,
            yes_fill_price=yes_fill,
            no_fill_price=no_fill,
            size_usdc=size_usdc,
            success=success,
            dry_run=self.dry_run,
            error=error_msg,
        )


# ---------------------------------------------------------------------------
# Arbitrage Bot
# ---------------------------------------------------------------------------

class ArbitrageBot:
    """
    Main orchestrator.

    Lifecycle
    ---------
    1. Connect to Polymarket CLOB WebSocket.
    2. Subscribe to YES and NO token order books for the target market.
    3. On every book update check for arbitrage using _evaluate_signal().
    4. When a signal is approved by RiskManager, execute via TradingEngine.
    5. Reconnect automatically on WebSocket drops.
    6. Fall back to REST polling if WebSocket fails repeatedly.

    Shutdown
    --------
    Send SIGINT / SIGTERM (Ctrl-C) for a clean shutdown that logs final stats.
    """

    def __init__(self, config: Config) -> None:
        self.config  = config
        self.risk    = RiskManager(
            max_position_usdc=config.max_position_usdc,
            min_net_profit_pct=config.min_net_profit_pct,
        )
        self.engine  = TradingEngine(config)
        self.yes_book = OrderBook(token_id=config.yes_token_id)
        self.no_book  = OrderBook(token_id=config.no_token_id)

        self._running          = False
        self._ws_reconnects    = 0
        self._signals_detected = 0
        self._trades_executed  = 0
        self._log              = logging.getLogger("poly.arb.bot")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        self._install_signal_handlers()

        mode = "DRY RUN" if self.config.dry_run else "LIVE TRADING"
        self._log.info(
            "=== Polymarket Arb Bot START | mode=%s | market=%s ===",
            mode,
            self.config.condition_id,
        )

        try:
            await self._ws_loop()
        finally:
            self._log.info("=== Bot stopped | stats=%s ===", self.risk.stats())

    # ------------------------------------------------------------------
    # WebSocket loop with reconnect + REST fallback
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Outer reconnection loop.  Falls back to REST polling after too many failures."""
        while self._running:
            try:
                await self._ws_session()
                self._ws_reconnects = 0   # reset on clean close
            except Exception as exc:
                if not self._running:
                    break
                self._ws_reconnects += 1
                self._log.warning(
                    "WebSocket error (attempt %d/%d): %s",
                    self._ws_reconnects,
                    self.config.ws_max_reconnect_attempts,
                    exc,
                )
                if self._ws_reconnects >= self.config.ws_max_reconnect_attempts:
                    self._log.error("Max WS reconnects reached.  Falling back to REST polling.")
                    await self._rest_poll_loop()
                    return
                await asyncio.sleep(self.config.ws_reconnect_delay_s)

    async def _ws_session(self) -> None:
        """Open a single WebSocket session and process messages until it closes."""
        uri = CLOB_WS_URL
        self._log.info("Connecting to %s …", uri)

        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:
            self._log.info("WebSocket connected.  Subscribing to order books …")
            await self._subscribe(ws)

            async for raw in ws:
                if not self._running:
                    break
                try:
                    self._handle_message(json.loads(raw))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    self._log.debug("Malformed WS message ignored: %s | raw=%s", exc, raw[:200])

    async def _subscribe(self, ws) -> None:
        """
        Send the market-data subscription for both outcome tokens.

        Polymarket CLOB WebSocket subscription format:
            channel: "market"  — public, no auth required for read-only data
            assets_ids: list of token IDs to watch
        """
        sub = {
            "type": "subscribe",
            "channel": "market",
            "assets_ids": [self.config.yes_token_id, self.config.no_token_id],
        }
        await ws.send(json.dumps(sub))
        self._log.debug("Subscribed: %s", sub)

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _handle_message(self, msg: dict) -> None:
        """Route incoming WebSocket messages to the appropriate handler."""
        evt = msg.get("event_type") or msg.get("type", "")

        if evt == "book":
            self._on_book_snapshot(msg)
        elif evt == "price_change":
            self._on_price_change(msg)
        elif evt in ("subscribed", "connected", "last_trade_price"):
            # Informational — log at DEBUG level only
            self._log.debug("WS info message: %s", msg)
        else:
            self._log.debug("Unhandled event_type=%s", evt)

    def _on_book_snapshot(self, msg: dict) -> None:
        asset_id = msg.get("asset_id", "")
        book     = self._book_for(asset_id)
        if book is None:
            return
        book.apply_snapshot(
            buys=msg.get("buys", []),
            sells=msg.get("sells", []),
        )
        self._log.debug(
            "Snapshot | token=%s…  best_ask=%s  best_bid=%s",
            asset_id[:8],
            book.best_ask(),
            book.best_bid(),
        )
        asyncio.get_event_loop().create_task(self._maybe_trade())

    def _on_price_change(self, msg: dict) -> None:
        asset_id = msg.get("asset_id", "")
        book     = self._book_for(asset_id)
        if book is None:
            return
        book.apply_changes(msg.get("changes", []))
        self._log.debug(
            "Delta | token=%s…  best_ask=%s",
            asset_id[:8],
            book.best_ask(),
        )
        asyncio.get_event_loop().create_task(self._maybe_trade())

    def _book_for(self, asset_id: str) -> Optional[OrderBook]:
        if asset_id == self.config.yes_token_id:
            return self.yes_book
        if asset_id == self.config.no_token_id:
            return self.no_book
        return None

    # ------------------------------------------------------------------
    # REST polling fallback
    # ------------------------------------------------------------------

    async def _rest_poll_loop(self) -> None:
        """
        Poll the CLOB REST API for order book snapshots when WebSocket is down.
        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        import requests  # already in requirements; intentionally local import

        self._log.info("REST polling mode | interval=%.1fs", self.config.poll_interval_s)
        session = requests.Session()
        session.headers["Accept"] = "application/json"

        while self._running:
            for token_id, book in [
                (self.config.yes_token_id, self.yes_book),
                (self.config.no_token_id,  self.no_book),
            ]:
                try:
                    data = await asyncio.to_thread(
                        lambda tid=token_id: session.get(
                            f"{CLOB_REST_URL}/book",
                            params={"token_id": tid},
                            timeout=5,
                        ).json()
                    )
                    book.apply_snapshot(
                        buys=data.get("bids", []),
                        sells=data.get("asks", []),
                    )
                except Exception as exc:
                    self._log.warning("REST poll error for %s: %s", token_id[:8], exc)

            await self._maybe_trade()
            await asyncio.sleep(self.config.poll_interval_s)

    # ------------------------------------------------------------------
    # Core arbitrage logic
    # ------------------------------------------------------------------

    def _evaluate_signal(self) -> Optional[ArbitrageSignal]:
        """
        Compute whether an arbitrage opportunity currently exists.

        Returns an ArbitrageSignal if profitable, otherwise None.
        """
        yes_ask_lvl = self.yes_book.best_ask()
        no_ask_lvl  = self.no_book.best_ask()

        if yes_ask_lvl is None or no_ask_lvl is None:
            return None  # incomplete order book — wait for more data

        yes_ask      = yes_ask_lvl.price
        no_ask       = no_ask_lvl.price
        combined     = yes_ask + no_ask
        net_profit   = PAYOUT_AFTER_FEE - combined - TOTAL_GAS_COST

        if combined == 0:
            return None

        net_profit_pct = net_profit / combined

        return ArbitrageSignal(
            yes_ask=yes_ask,
            no_ask=no_ask,
            combined_cost=combined,
            payout=PAYOUT_AFTER_FEE,
            gas_cost=TOTAL_GAS_COST,
            net_profit=net_profit,
            net_profit_pct=net_profit_pct,
        )

    async def _maybe_trade(self) -> None:
        """
        Called after every order book update.  Evaluates signal, runs risk
        checks, sizes the position, and fires the trade if everything passes.
        """
        signal = self._evaluate_signal()

        if signal is None or signal.net_profit <= 0:
            return

        self._signals_detected += 1
        self._log.info("%s", signal)

        approved, reason = self.risk.approve_signal(signal)
        if not approved:
            self._log.info("Signal rejected by RiskManager: %s", reason)
            return

        size_usdc = self.risk.compute_position_size(signal, self.yes_book, self.no_book)
        if size_usdc <= 0:
            self._log.warning("Zero position size computed — skipping (insufficient liquidity)")
            return

        self.risk.open_trade()
        try:
            result = await self.engine.execute_arbitrage(
                signal=signal,
                yes_token_id=self.config.yes_token_id,
                no_token_id=self.config.no_token_id,
                size_usdc=size_usdc,
            )
        finally:
            self.risk.close_trade(result if "result" in dir() else TradeResult(
                signal=signal, yes_order_id=None, no_order_id=None,
                yes_fill_price=None, no_fill_price=None,
                size_usdc=size_usdc, success=False, dry_run=self.config.dry_run,
                error="Exception before result was set"))

        self._trades_executed += 1
        self._log.info(
            "Trade #%d complete | success=%s | dry_run=%s | stats=%s",
            self._trades_executed,
            result.success,
            result.dry_run,
            self.risk.stats(),
        )

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_stop_signal)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler for all sigs
                pass

    def _handle_stop_signal(self) -> None:
        self._log.info("Shutdown signal received.  Stopping …")
        self._running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Load config from environment / .env and start the bot.

    Command-line flags
    ------------------
    --dry-run   Force dry-run mode regardless of DRY_RUN env var.
    --live      Force live trading mode  (requires all POLY_* env vars).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Polymarket intra-market arbitrage bot"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true", help="Simulate trades (no real orders)")
    mode_group.add_argument("--live",    action="store_true", help="Execute real orders on Polygon")
    args = parser.parse_args()

    # CLI flags override env var
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"
    elif args.live:
        os.environ["DRY_RUN"] = "false"

    try:
        config = Config()
    except EnvironmentError as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)

    log.info(
        "Config loaded | condition_id=%s | dry_run=%s | max_position=%.2f USDC | "
        "min_profit=%.3f%%",
        config.condition_id,
        config.dry_run,
        config.max_position_usdc,
        float(config.min_net_profit_pct) * 100,
    )

    bot = ArbitrageBot(config)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
