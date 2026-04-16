# Polymarket Intra-Market Arbitrage Bot

## Overview

Built a fully async Python arbitrage bot targeting **intra-market arbitrage** on Polymarket's binary prediction markets using `py-clob-client` and WebSockets.

In a binary market, one outcome (YES or NO) resolves to **$1.00** and the other to **$0.00**. If you buy both tokens, you are guaranteed to collect $1.00. After Polymarket's 2% fee on winnings and estimated Polygon gas costs, the guaranteed payout is **$0.94**. An arb opportunity exists whenever the combined ask price of both tokens falls below that threshold.

$$\text{net profit} = \underbrace{0.98}_{\text{payout after 2\% fee}} - (\text{YES ask} + \text{NO ask}) - \underbrace{\$0.04}_{\text{gas}} > 0$$

---

## Files Created

| File | Purpose |
|---|---|
| `prediction_markets/polymarket_arb.py` | Main bot — strategy, risk, execution |
| `prediction_markets/test_polymarket_arb.py` | 34-test unit test suite (no credentials needed) |

---

## Architecture

### `Config` (dataclass)
Loads all runtime parameters from environment variables via `python-dotenv`. Fails fast with a descriptive error if any required variable is missing. Supports `--dry-run` and `--live` CLI flags that override the `DRY_RUN` env var.

Required env vars:

```
POLY_PRIVATE_KEY      Polygon wallet private key (no 0x prefix)
POLY_API_KEY          Polymarket CLOB API key
POLY_API_SECRET       Polymarket CLOB API secret
POLY_PASSPHRASE       Polymarket CLOB API passphrase
CONDITION_ID          Target market condition_id
YES_TOKEN_ID          CLOB asset_id for the YES token
NO_TOKEN_ID           CLOB asset_id for the NO token
DRY_RUN               "true" (default) or "false"
MAX_POSITION_USDC     USDC cap per trade (default: 100)
MIN_NET_PROFIT_PCT    Minimum profit % to trigger (default: 0.5)
```

---

### `OrderBook` (dataclass)
Per-token order book with two update modes:

- **`apply_snapshot(buys, sells)`** — replaces the entire book state (used on WS `book` events)
- **`apply_changes(changes)`** — applies incremental price-level deltas (used on `price_change` events). A size of `"0"` removes the level.
- **`available_at_or_below(max_price)`** — returns total USDC liquidity at or below a given price, used for position sizing.

---

### `RiskManager`
Pre-trade and post-trade guardrails:

| Check | Guard |
|---|---|
| Minimum net profit % | Rejects signals below the configured threshold |
| Daily loss limit | Stops all trading if realised losses exceed `max_daily_loss_usdc` |
| Concurrent trade cap | Limits simultaneous open arb positions |
| Position sizing | Caps trade size at `min(max_position_usdc, bottleneck_leg_liquidity)` |

Tracks P&L, win rate, and trade history across the session. Exposes `reset_daily_pnl()` for UTC midnight resets.

---

### `TradingEngine`
Wraps `py-clob-client` for async order execution:

- All blocking CLOB calls dispatched via `asyncio.to_thread()` — the event loop is never stalled.
- Orders are submitted as **Fill-Or-Kill (FOK)** market buys — atomic fills only, no partial open orders left on the book.
- **Both legs are fired concurrently** via `asyncio.gather()` to minimise the window between the YES and NO fills.
- In dry-run mode, all order logic is intercepted before any network call and a synthetic fill is returned.

---

### `ArbitrageBot`
Main async orchestrator:

```
WS connect → subscribe(YES, NO) → book update → _evaluate_signal()
    → RiskManager.approve_signal()
    → RiskManager.compute_position_size()
    → TradingEngine.execute_arbitrage()  [both legs concurrent]
    → RiskManager.close_trade()
```

**Resilience:**
- Auto-reconnects on WebSocket drops (up to 10 attempts with 3 s back-off)
- Falls back to REST polling (`/book` endpoint) if all reconnect attempts fail
- Handles `SIGINT`/`SIGTERM` gracefully, logging final stats before exit

---

## Test Suite

34 unit tests across 6 suites, all passing. No credentials, network, or `py-clob-client` install required.

```
TestOrderBook          7 tests — snapshot, delta, liquidity depth
TestRiskManager        9 tests — all approval guards, position sizing, P&L
TestArbitrageSignal    5 tests — profit math, boundary, string repr
TestEvaluateSignal     4 tests — empty/partial book, profitable/unprofitable detection
TestTradingEngineDryRun 4 tests — dry-run order placement, concurrent legs
TestConfigValidation   5 tests — missing env var error, type parsing
```

Run:
```bash
PYTHONPATH=. python3 prediction_markets/test_polymarket_arb.py
```

---

## Usage

```bash
# 1. Install deps
pip install py-clob-client websockets python-dotenv

# 2. Fill in credentials in prediction_markets/.env
#    (stubs already appended — search "Polymarket CLOB")

# 3. Get token IDs for a market
#    from prediction_markets import PolymarketClient
#    m = PolymarketClient().get_market("<condition_id>")
#    yes_id, no_id = m["clobTokenIds"]

# 4. Dry-run (safe — no real orders)
PYTHONPATH=. python3 -m prediction_markets.polymarket_arb --dry-run

# 5. Live trading
PYTHONPATH=. python3 -m prediction_markets.polymarket_arb --live
```

---

## Security Notes

- The private key is loaded once from the environment and passed directly to `py-clob-client` — it is never logged.
- `DRY_RUN=true` is the default; live trading requires an explicit opt-in.
- All user-supplied env vars are validated at startup before any network connection is opened.
