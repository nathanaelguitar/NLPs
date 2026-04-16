# Cross-Market Prediction Market Arbitrage Constraint Engine

## Overview

A pure-Python scanner that evaluates **logical pricing relationships** across a collection of binary prediction markets and surfaces violations as ranked, actionable arbitrage signals.

In prediction markets, prices are probabilities. When markets are logically related — one event implies another, two events are mutually exclusive, an OR event can't be cheaper than its components — the prices must satisfy mathematical constraints. When they don't, a risk-free (or near-risk-free) trade exists.

This engine formalises those constraints, evaluates them against a market snapshot, and ranks every violation by how much money is available to capture.

---

## File

`prediction_markets/cross_market_arb.py` — stdlib only, no external dependencies.

---

## Architecture

```
DEMO_MARKETS  ──► ConstraintEngine.load_markets()
                          │
build_demo_constraints()──► ConstraintEngine.register_many()
                          │
                     evaluate()
                          │
               List[Violation]  (ranked by severity)
                          │
                     Reporter.render()
                          │
                     stdout report
```

All layers are cleanly separated. The engine and models have no I/O dependencies and can be imported headlessly into tests, pipelines, or async bots.

---

## Data Models

### `Market` _(frozen dataclass)_
Immutable price snapshot for a single binary prediction market.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique market identifier |
| `question` | `str` | Human-readable market question |
| `yes_price` | `Decimal` | Best-ask price for the YES token (0–1) |
| `no_price` | `Decimal` | Best-ask price for the NO token (0–1) |
| `volume_usdc` | `Decimal` | Total traded volume |
| `liquidity_usdc` | `Decimal` | On-book depth (used for arb sizing) |
| `tags` | `Tuple[str, ...]` | Optional category labels |

### `Constraint` _(dataclass)_
A named, typed pricing rule across one or more markets. The `check(markets)` callable returns a `Violation` when the rule is breached, or `None` when it holds.

### `Violation` _(frozen dataclass)_
A fired constraint result.

| Field | Description |
|---|---|
| `deviation` | Raw probability-space magnitude of the breach |
| `severity` | Deviation × log-liquidity — primary ranking key |
| `arb_profit_usdc` | Conservative USDC profit estimate at 30% capture rate |
| `detail` | Human-readable explanation with suggested trade direction |

### `ConstraintType` _(Enum)_
`COMPLEMENT` · `IMPLICATION` · `CONJUNCTION` · `DISJUNCTION_LO` · `DISJUNCTION_HI` · `MUT_EXCL` · `PARTITION`

---

## Constraint Types

### Complement Pricing
Every binary market must satisfy $P(\text{YES}) + P(\text{NO}) \approx 1.0$.

| Breach | Trade |
|---|---|
| Sum < 1 | Buy YES + Buy NO for combined cost; collect \$1.00 at resolution |
| Sum > 1 | Sell YES + Sell NO (requires shorting); collect the excess |

---

### Implication
If event $A \subseteq B$ (A is more specific than B), then $P(B) \geq P(A)$.

**Example:** "Fed hikes in Q1" implies "Fed hikes in H1" implies "Fed hikes in 2026."

$$P(\text{Q1 hike}) \leq P(\text{H1 hike}) \leq P(\text{2026 hike})$$

**Trade:** Buy the underpriced, more general event (B); fade the overpriced specific event (A).

---

### Conjunction Bound
A joint event cannot be more likely than either of its components.

$$P(A \wedge B) \leq \min(P(A),\, P(B))$$

**Trade:** Sell the overpriced AND market; hedge with NO positions on A or B.

---

### Disjunction Lower Bound
An OR event must be at least as likely as the most probable of its components.

$$P(A \vee B) \geq \max(P(A),\, P(B))$$

**Trade:** Buy the underpriced OR market.

---

### Disjunction Upper Bound
An OR event cannot exceed the sum of its components (inclusion-exclusion).

$$P(A \vee B) \leq P(A) + P(B)$$

**Trade:** Sell the overpriced OR market.

---

### Mutual Exclusivity
For a set of mutually exclusive outcomes (only one can occur), prices must sum to at most 1.

$$\sum_i P(i) \leq 1.0$$

**Trade:** Sell YES on every outcome; collect $\sum P(i)$, pay out at most \$1.00.

---

### Exhaustive Partition
For an exhaustive partition of the outcome space (exactly one outcome occurs), prices must sum to exactly 1.

$$\sum_i P(i) = 1.0 \pm \epsilon$$

| Breach | Trade |
|---|---|
| Sum < 1 | Buy all outcomes for the combined cost; collect \$1.00 at resolution |
| Sum > 1 | Sell all outcomes; collect the excess over \$1.00 |

---

## Severity Ranking

$$\text{severity} = \Delta_{\text{prob}} \times \log_{10}(1 + \Sigma\,\text{liquidity})$$

Balances *how wrong* the price is against *how much capital is at stake*. `log10` prevents one giant illiquid market from dominating smaller, liquid violations.

Arb profit estimate:

$$\text{arb profit} \approx \Delta_{\text{prob}} \times \min(\text{leg liquidities}) \times 0.30$$

The 30% capture rate is conservative — assumes partial fills due to slippage and competing arb bots.

---

## Demo Dataset Output

20 intentionally-mispriced markets across 10 constraints. All 7 constraint types fire at least once.

```
════════════════════════════════════════════════════════════════════════════════
  POLYMARKET CROSS-MARKET CONSTRAINT ENGINE — VIOLATION SCAN
  Markets scanned : 20  |  Constraints checked : 10  |  Violations : 10
════════════════════════════════════════════════════════════════════════════════
  #1   SEVERITY 1.0753  │  IMPLICATION         │  Δ = 0.25   arb ≈  $600
  #2   SEVERITY 0.8214  │  MUTUAL EXCLUSIVITY  │  Δ = 0.20   arb ≈  $150
  #3   SEVERITY 0.7365  │  IMPLICATION         │  Δ = 0.17   arb ≈  $484
  #4   SEVERITY 0.6487  │  DISJUNCTION LO      │  Δ = 0.14   arb ≈  $252
  #5   SEVERITY 0.6017  │  CONJUNCTION         │  Δ = 0.13   arb ≈  $312
  #6   SEVERITY 0.5688  │  COMPLEMENT          │  Δ = 0.12   arb ≈ $1,980
  #7   SEVERITY 0.5509  │  PARTITION           │  Δ = 0.12   arb ≈  $360
  #8   SEVERITY 0.5337  │  COMPLEMENT          │  Δ = 0.12   arb ≈ $1,008
  #9   SEVERITY 0.4663  │  DISJUNCTION HI      │  Δ = 0.10   arb ≈  $450
  #10  SEVERITY 0.3394  │  IMPLICATION         │  Δ = 0.08   arb ≈  $192

  Total estimated arb opportunity .............. $5,788.50
```

---

## Running

```bash
# From the project root
PYTHONPATH=. python3 prediction_markets/cross_market_arb.py

# or as a module
python3 -m prediction_markets.cross_market_arb
```

---

## Extending with Live Polymarket Data

The engine is deliberately decoupled from any data source. Swap in the existing `PolymarketClient` like this:

```python
from prediction_markets.polymarket_client import PolymarketClient
from prediction_markets.cross_market_arb import (
    ConstraintEngine, Market, Reporter,
    complement_constraint, implication_constraint,
    build_demo_constraints,
)
from decimal import Decimal

client = PolymarketClient()
raw    = client.get_all_active_markets()

def to_market(m: dict) -> Market:
    token_ids = m.get("clobTokenIds", ["", ""])
    prices    = client.get_prices(token_ids[0]) if token_ids else {}
    return Market(
        id             = m["conditionId"],
        question       = m["question"],
        yes_price      = Decimal(str(prices.get("yes", 0.5))),
        no_price       = Decimal(str(prices.get("no",  0.5))),
        volume_usdc    = Decimal(str(m.get("volume", 0))),
        liquidity_usdc = Decimal(str(m.get("liquidity", 0))),
    )

engine = ConstraintEngine()
engine.load_markets(to_market(m) for m in raw)

# Add complement check for every market in the feed
for mid in engine.market_ids():
    engine.register(complement_constraint(mid))

# Add any known logical relationships
engine.register(implication_constraint("fed_q1_hike", "fed_2026_hike"))

violations = engine.violations_by_severity()
Reporter.render(violations, engine.market_count(), engine.constraint_count())
```

---

## Adding New Constraint Types

1. Write a factory function that returns a `Constraint` with a `check` closure.
2. The closure receives `Dict[str, Market]` and returns `Optional[Violation]`.
3. Register with `engine.register(my_constraint(...))`.
4. No other changes required — the engine, reporter, and ranking are generic.

```python
def my_custom_constraint(market_id: str, threshold: Decimal) -> Constraint:
    def check(markets):
        m = markets.get(market_id)
        if m is None:
            return None
        deviation = (m.yes_price - threshold).copy_abs()
        if deviation < Decimal("0.05"):
            return None
        # build and return a Violation ...

    return Constraint(
        id         = f"custom:{market_id}",
        type       = ConstraintType.COMPLEMENT,   # closest existing type, or add a new enum value
        description= f"Custom rule on {market_id}",
        market_ids = (market_id,),
        check      = check,
    )
```
