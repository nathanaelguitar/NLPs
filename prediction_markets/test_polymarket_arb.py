"""
Unit tests for polymarket_arb.py
=================================
Tests exercise every core class in isolation — no network, no credentials,
no WebSocket needed.  Run:

    python -m prediction_markets.test_polymarket_arb
    # or
    python prediction_markets/test_polymarket_arb.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
import unittest
from decimal import Decimal
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Patch env vars BEFORE importing the module so Config() doesn't blow up
# ---------------------------------------------------------------------------
_FAKE_ENV = {
    "POLY_PRIVATE_KEY":   "aabbcc" * 10,
    "POLY_API_KEY":       "test-api-key",
    "POLY_API_SECRET":    "test-api-secret",
    "POLY_PASSPHRASE":    "test-passphrase",
    "CONDITION_ID":       "0xdeadbeef" * 4,
    "YES_TOKEN_ID":       "yes_token_123",
    "NO_TOKEN_ID":        "no_token_456",
    "DRY_RUN":            "true",
    "MAX_POSITION_USDC":  "200",
    "MIN_NET_PROFIT_PCT": "0.5",
}

# Set env before any imports so load_dotenv() / os.getenv() picks them up
for k, v in _FAKE_ENV.items():
    os.environ[k] = v

# Stub websockets so we can import the module even if it's not installed
import types
if "websockets" not in sys.modules:
    ws_stub = types.ModuleType("websockets")
    ws_stub.connect = None  # never actually called in tests
    sys.modules["websockets"] = ws_stub

from prediction_markets.polymarket_arb import (  # noqa: E402
    PAYOUT_AFTER_FEE,
    TOTAL_GAS_COST,
    ArbitrageBot,
    ArbitrageSignal,
    Config,
    OrderBook,
    RiskManager,
    TradingEngine,
    TradeResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_book(token_id: str, asks: dict, bids: Optional[dict] = None) -> OrderBook:
    """Convenience: build an OrderBook with pre-set asks/bids (price->size)."""
    book = OrderBook(token_id=token_id)
    book.asks = {Decimal(str(p)): Decimal(str(s)) for p, s in asks.items()}
    book.bids = {Decimal(str(p)): Decimal(str(s)) for p, s in (bids or {}).items()}
    return book


def _make_signal(yes_ask: str, no_ask: str) -> ArbitrageSignal:
    y = Decimal(yes_ask)
    n = Decimal(no_ask)
    combined = y + n
    profit   = PAYOUT_AFTER_FEE - combined - TOTAL_GAS_COST
    return ArbitrageSignal(
        yes_ask=y,
        no_ask=n,
        combined_cost=combined,
        payout=PAYOUT_AFTER_FEE,
        gas_cost=TOTAL_GAS_COST,
        net_profit=profit,
        net_profit_pct=profit / combined if combined else Decimal("0"),
    )


def _make_config(**overrides) -> Config:
    """Build a Config, optionally overriding specific fields."""
    cfg = Config()
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


# ============================================================================
# Test suites
# ============================================================================


class TestOrderBook(unittest.TestCase):
    """Tests for OrderBook snapshot / delta / query methods."""

    def setUp(self):
        self.book = OrderBook(token_id="tok_abc")

    def test_empty_book_has_no_best_ask(self):
        self.assertIsNone(self.book.best_ask())
        self.assertIsNone(self.book.best_bid())

    def test_snapshot_populates_book(self):
        self.book.apply_snapshot(
            buys=[{"price": "0.48", "size": "50"}],
            sells=[{"price": "0.52", "size": "30"}, {"price": "0.54", "size": "10"}],
        )
        self.assertAlmostEqual(float(self.book.best_ask().price), 0.52)
        self.assertAlmostEqual(float(self.book.best_bid().price), 0.48)

    def test_snapshot_replaces_previous_state(self):
        self.book.apply_snapshot(
            buys=[{"price": "0.30", "size": "100"}],
            sells=[{"price": "0.70", "size": "5"}],
        )
        self.book.apply_snapshot(
            buys=[{"price": "0.45", "size": "20"}],
            sells=[{"price": "0.55", "size": "60"}],
        )
        self.assertAlmostEqual(float(self.book.best_ask().price), 0.55)
        self.assertAlmostEqual(float(self.book.best_bid().price), 0.45)
        # old level from first snapshot must be gone
        self.assertNotIn(Decimal("0.70"), self.book.asks)

    def test_delta_update_adds_new_level(self):
        self.book.asks = {Decimal("0.52"): Decimal("30")}
        self.book.apply_changes([{"price": "0.50", "size": "15", "side": "sell"}])
        self.assertIn(Decimal("0.50"), self.book.asks)
        self.assertAlmostEqual(float(self.book.best_ask().price), 0.50)

    def test_delta_update_removes_level_on_zero_size(self):
        self.book.asks = {Decimal("0.52"): Decimal("30"), Decimal("0.54"): Decimal("10")}
        self.book.apply_changes([{"price": "0.52", "size": "0", "side": "sell"}])
        self.assertNotIn(Decimal("0.52"), self.book.asks)
        self.assertAlmostEqual(float(self.book.best_ask().price), 0.54)

    def test_available_at_or_below(self):
        self.book.asks = {
            Decimal("0.48"): Decimal("100"),   # 0.48 * 100 = 48.0
            Decimal("0.50"): Decimal("50"),    # 0.50 * 50  = 25.0
            Decimal("0.55"): Decimal("200"),   # excluded — above cap
        }
        liq = self.book.available_at_or_below(Decimal("0.50"))
        self.assertAlmostEqual(float(liq), 73.0)

    def test_side_case_buy_delta(self):
        """BUY deltas should affect bids, not asks."""
        self.book.bids = {}
        self.book.apply_changes([{"price": "0.45", "size": "20", "side": "buy"}])
        self.assertIn(Decimal("0.45"), self.book.bids)
        self.assertEqual(len(self.book.asks), 0)


class TestRiskManager(unittest.TestCase):
    """Tests for RiskManager approval and P&L tracking."""

    def setUp(self):
        self.rm = RiskManager(
            max_position_usdc=Decimal("100"),
            min_net_profit_pct=Decimal("0.005"),   # 0.5 %
            max_daily_loss_usdc=Decimal("50"),
            max_concurrent_trades=2,
        )

    def _arb_signal(self, net_pct: float) -> ArbitrageSignal:
        combined = Decimal("0.90")
        profit   = combined * Decimal(str(net_pct))
        return ArbitrageSignal(
            yes_ask=Decimal("0.45"),
            no_ask=Decimal("0.45"),
            combined_cost=combined,
            payout=PAYOUT_AFTER_FEE,
            gas_cost=TOTAL_GAS_COST,
            net_profit=profit,
            net_profit_pct=Decimal(str(net_pct)),
        )

    def test_approve_valid_signal(self):
        approved, reason = self.rm.approve_signal(self._arb_signal(0.01))
        self.assertTrue(approved)
        self.assertEqual(reason, "OK")

    def test_reject_below_min_profit(self):
        approved, reason = self.rm.approve_signal(self._arb_signal(0.001))
        self.assertFalse(approved)
        self.assertIn("minimum", reason)

    def test_reject_daily_loss_exceeded(self):
        self.rm._daily_pnl = Decimal("-51")
        approved, reason = self.rm.approve_signal(self._arb_signal(0.02))
        self.assertFalse(approved)
        self.assertIn("Daily loss", reason)

    def test_reject_max_concurrent_trades(self):
        self.rm._open_trade_count = 2
        approved, reason = self.rm.approve_signal(self._arb_signal(0.02))
        self.assertFalse(approved)
        self.assertIn("concurrent", reason)

    def test_position_size_capped_by_max(self):
        signal = self._arb_signal(0.01)
        yes_book = _make_book("yes", asks={"0.45": 1000})
        no_book  = _make_book("no",  asks={"0.45": 1000})
        size = self.rm.compute_position_size(signal, yes_book, no_book)
        self.assertEqual(size, Decimal("100"))   # capped at max_position_usdc

    def test_position_size_limited_by_liquidity(self):
        signal = self._arb_signal(0.01)
        # Only $30 of YES liquidity available at 0.45 (0.45 * 66.67 ≈ 30)
        yes_book = _make_book("yes", asks={"0.45": "10"})   # 0.45 * 10 = 4.5
        no_book  = _make_book("no",  asks={"0.45": "1000"})
        size = self.rm.compute_position_size(signal, yes_book, no_book)
        # bottleneck is yes_book: 0.45 * 10 = 4.5
        self.assertLess(float(size), 100)
        self.assertAlmostEqual(float(size), 4.5)

    def test_stats_initial(self):
        s = self.rm.stats()
        self.assertEqual(s["total_trades"], 0)
        self.assertEqual(s["daily_pnl_usdc"], 0)

    def test_reset_daily_pnl(self):
        self.rm._daily_pnl = Decimal("-30")
        self.rm.reset_daily_pnl()
        self.assertEqual(self.rm._daily_pnl, Decimal("0"))

    def test_open_close_trade_counter(self):
        self.rm.open_trade()
        self.rm.open_trade()
        self.assertEqual(self.rm._open_trade_count, 2)
        dummy_result = TradeResult(
            signal=self._arb_signal(0.01),
            yes_order_id=None, no_order_id=None,
            yes_fill_price=None, no_fill_price=None,
            size_usdc=Decimal("10"),
            success=False, dry_run=True,
        )
        self.rm.close_trade(dummy_result)
        self.assertEqual(self.rm._open_trade_count, 1)


class TestArbitrageSignal(unittest.TestCase):
    """Tests for the core arb math."""

    def test_profitable_signal(self):
        """Combined ask well below payout -> positive profit."""
        # YES=0.44, NO=0.44 -> combined=0.88; payout=0.98; gas=0.04 -> profit=0.06
        sig = _make_signal("0.44", "0.44")
        self.assertGreater(sig.net_profit, 0)
        self.assertGreater(sig.net_profit_pct, 0)

    def test_unprofitable_signal_at_par(self):
        """Combined ask at exactly $0.98 (payout) still unprofitable after gas."""
        sig = _make_signal("0.49", "0.49")   # combined 0.98 - 0.04 gas = -0.04
        self.assertLess(sig.net_profit, 0)

    def test_unprofitable_overpriced_book(self):
        """Combined > $1 is always losing."""
        sig = _make_signal("0.55", "0.55")
        self.assertLess(sig.net_profit, 0)

    def test_boundary_exact_payout_minus_gas(self):
        """Net profit is zero when combined = payout - gas."""
        breakeven = PAYOUT_AFTER_FEE - TOTAL_GAS_COST
        half = breakeven / 2
        sig = _make_signal(str(half), str(half))
        self.assertAlmostEqual(float(sig.net_profit), 0.0, places=10)

    def test_str_representation_contains_key_fields(self):
        sig = _make_signal("0.44", "0.44")
        s = str(sig)
        self.assertIn("ARB SIGNAL", s)
        self.assertIn("YES ask", s)
        self.assertIn("NO ask", s)
        self.assertIn("net_profit", s)


class TestEvaluateSignal(unittest.TestCase):
    """Tests for ArbitrageBot._evaluate_signal() via a headless bot instance."""

    def setUp(self):
        cfg = _make_config()
        self.bot = ArbitrageBot(cfg)

    def test_returns_none_when_books_empty(self):
        result = self.bot._evaluate_signal()
        self.assertIsNone(result)

    def test_returns_none_when_only_yes_book_populated(self):
        self.bot.yes_book = _make_book("yes", asks={"0.44": "100"})
        result = self.bot._evaluate_signal()
        self.assertIsNone(result)

    def test_detects_profitable_opportunity(self):
        self.bot.yes_book = _make_book("yes", asks={"0.40": "200"})
        self.bot.no_book  = _make_book("no",  asks={"0.40": "200"})
        sig = self.bot._evaluate_signal()
        self.assertIsNotNone(sig)
        self.assertGreater(sig.net_profit, 0)

    def test_returns_signal_even_when_unprofitable(self):
        """_evaluate_signal always returns a signal; caller filters on profit > 0."""
        self.bot.yes_book = _make_book("yes", asks={"0.55": "10"})
        self.bot.no_book  = _make_book("no",  asks={"0.55": "10"})
        sig = self.bot._evaluate_signal()
        self.assertIsNotNone(sig)
        self.assertLess(sig.net_profit, 0)


class TestTradingEngineDryRun(unittest.IsolatedAsyncioTestCase):
    """Tests for TradingEngine in dry-run mode — no network calls made."""

    def setUp(self):
        self.cfg = _make_config(dry_run=True)
        # In dry-run mode _build_client returns None; that's fine
        self.engine = TradingEngine(self.cfg)

    async def test_place_market_order_returns_filled(self):
        result = await self.engine.place_market_order("tok_abc123", Decimal("50"), "TEST")
        self.assertEqual(result["status"], "filled")
        self.assertIn("order_id", result)

    async def test_order_id_contains_token_prefix(self):
        result = await self.engine.place_market_order("tok_abc123", Decimal("10"))
        self.assertIn("tok_abc1", result["order_id"])

    async def test_execute_arbitrage_both_legs_filled(self):
        signal = _make_signal("0.44", "0.44")
        result = await self.engine.execute_arbitrage(
            signal=signal,
            yes_token_id="yes_token_123",
            no_token_id="no_token_456",
            size_usdc=Decimal("50"),
        )
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertTrue(result.dry_run)

    async def test_execute_arbitrage_result_has_correct_size(self):
        signal = _make_signal("0.44", "0.44")
        result = await self.engine.execute_arbitrage(
            signal=signal,
            yes_token_id="yes_token_123",
            no_token_id="no_token_456",
            size_usdc=Decimal("75"),
        )
        self.assertEqual(result.size_usdc, Decimal("75"))


class TestConfigValidation(unittest.TestCase):
    """Tests that Config raises EnvironmentError for missing required vars."""

    def test_missing_required_env_raises(self):
        original = os.environ.pop("POLY_PRIVATE_KEY", None)
        try:
            with self.assertRaises(EnvironmentError) as ctx:
                Config()
            self.assertIn("POLY_PRIVATE_KEY", str(ctx.exception))
        finally:
            if original is not None:
                os.environ["POLY_PRIVATE_KEY"] = original

    def test_dry_run_defaults_to_true(self):
        os.environ["DRY_RUN"] = "true"
        cfg = Config()
        self.assertTrue(cfg.dry_run)

    def test_dry_run_false_when_explicitly_set(self):
        os.environ["DRY_RUN"] = "false"
        cfg = Config()
        self.assertFalse(cfg.dry_run)
        os.environ["DRY_RUN"] = "true"  # reset

    def test_max_position_parsed_correctly(self):
        os.environ["MAX_POSITION_USDC"] = "250"
        cfg = Config()
        self.assertEqual(cfg.max_position_usdc, Decimal("250"))
        os.environ["MAX_POSITION_USDC"] = "200"  # reset

    def test_min_net_profit_pct_converted_to_fraction(self):
        os.environ["MIN_NET_PROFIT_PCT"] = "1.0"   # 1 % → 0.01 fraction
        cfg = Config()
        self.assertAlmostEqual(float(cfg.min_net_profit_pct), 0.01)
        os.environ["MIN_NET_PROFIT_PCT"] = "0.5"   # reset


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestOrderBook,
        TestRiskManager,
        TestArbitrageSignal,
        TestEvaluateSignal,
        TestTradingEngineDryRun,
        TestConfigValidation,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
