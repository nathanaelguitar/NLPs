"""
Cross-Market Prediction Market Arbitrage Constraint Engine
==========================================================
Scans a collection of prediction markets for pricing violations that arise
from logical relationships between markets.  When prices violate a constraint,
mispriced probability can be bought or sold against a theoretically correct
counterpart, extracting risk-free (or near-risk-free) profit.

Constraint types implemented
-----------------------------
  COMPLEMENT      YES_price + NO_price ≈ 1.0  (per-market)
  IMPLICATION     If A ⊆ B then P(B) ≥ P(A)   (e.g. Q1 hike ⊆ H1 hike)
  CONJUNCTION     P(A ∧ B) ≤ min(P(A), P(B))
  DISJUNCTION_LO  P(A ∨ B) ≥ max(P(A), P(B))
  DISJUNCTION_HI  P(A ∨ B) ≤ P(A) + P(B)
  MUT_EXCL        Σ mutually-exclusive P(i) ≤ 1.0
  PARTITION       Σ exhaustive-partition P(i) = 1.0

Architecture
------------
  Market            Immutable price snapshot (yes/no prices, volume, liquidity)
  ConstraintType    Enum of supported constraint flavours
  Violation         Fired constraint result with deviation, severity, arb estimate
  Constraint        Definition with a lazy check(markets) callable
  ConstraintEngine  Market + constraint registry; designed for live-data injection
  Reporter          Console rendering — fully separated from business logic
  Factories         complement_constraint(), implication_constraint(), …
  DEMO_MARKETS      Intentionally-mispriced in-memory dataset
  build_demo_constraints()  Wires all constraints over DEMO_MARKETS

Running
-------
    python prediction_markets/cross_market_arb.py
    # or
    PYTHONPATH=. python -m prediction_markets.cross_market_arb
"""

from __future__ import annotations

import math
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# |YES + NO − 1| threshold; absorbs Polymarket 2% fee + typical bid-ask spread
COMPLEMENT_TOLERANCE: Decimal = Decimal("0.03")

# Fraction of on-book liquidity assumed capturable per arb trade (conservative)
ARB_CAPTURE_RATE: Decimal = Decimal("0.30")

# Tolerance for exhaustive-partition sum-to-one check
PARTITION_TOLERANCE: Decimal = Decimal("0.03")

# Tolerance for mutual-exclusivity sum ≤ 1 check
MUT_EXCL_TOLERANCE: Decimal = Decimal("0.01")


# ─────────────────────────────────────────────────────────────────────────────
# Core data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Market:
    """
    Immutable snapshot of a single binary prediction market.

    yes_price : best-ask price for the YES outcome token  (0.0 – 1.0)
    no_price  : best-ask price for the NO  outcome token  (0.0 – 1.0)

    In a frictionless, fee-free market these sum to exactly 1.0.
    Structural violations occur when this sum deviates beyond tolerance,
    or when logical relationships between markets are breached.
    """

    id: str
    question: str
    yes_price: Decimal
    no_price: Decimal
    volume_usdc: Decimal
    liquidity_usdc: Decimal
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name, val in (("yes_price", self.yes_price), ("no_price", self.no_price)):
            if not (Decimal("0") <= val <= Decimal("1")):
                raise ValueError(f"Market '{self.id}': {name}={val} outside [0, 1]")

    @property
    def complement_sum(self) -> Decimal:
        return self.yes_price + self.no_price


class ConstraintType(Enum):
    COMPLEMENT     = "Complement Pricing"
    IMPLICATION    = "Implication"
    CONJUNCTION    = "Conjunction Bound"
    DISJUNCTION_LO = "Disjunction Lower Bound"
    DISJUNCTION_HI = "Disjunction Upper Bound"
    MUT_EXCL       = "Mutual Exclusivity"
    PARTITION      = "Exhaustive Partition"


@dataclass(frozen=True)
class Violation:
    """
    A constraint instance that fired — prices are inconsistent.

    severity        : deviation weighted by log-liquidity; primary ranking key
    deviation       : raw probability-space magnitude of the breach  (always > 0)
    arb_profit_usdc : conservative USDC profit estimate at ARB_CAPTURE_RATE
    detail          : human-readable explanation and suggested trade direction
    """

    constraint_id: str
    constraint_type: ConstraintType
    description: str
    market_ids: Tuple[str, ...]
    deviation: Decimal
    severity: Decimal
    arb_profit_usdc: Decimal
    detail: str

    def __lt__(self, other: "Violation") -> bool:
        return self.severity < other.severity


@dataclass
class Constraint:
    """
    A named, typed pricing relationship across one or more markets.

    check(market_snapshot) → Optional[Violation]
    Returns a Violation when the constraint is violated, else None.
    Missing market IDs always return None (graceful degradation).
    """

    id: str
    type: ConstraintType
    description: str
    market_ids: Tuple[str, ...]
    check: Callable[[Dict[str, Market]], Optional[Violation]]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _severity(deviation: Decimal, *liquidities: Decimal) -> Decimal:
    """
    severity = deviation × log10(1 + Σ liquidities)

    Balances the probability-space magnitude of the breach against the
    real capital at stake.  log10 prevents large illiquid markets dominating.
    """
    total = sum(liquidities, Decimal("0"))
    log_l = Decimal(str(math.log10(1.0 + float(total))))
    return (deviation * log_l).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _arb_usdc(deviation: Decimal, *liquidities: Decimal) -> Decimal:
    """
    Conservative arb-profit estimate.

      profit ≈ deviation × min(available liquidity) × ARB_CAPTURE_RATE

    Uses min-liquidity as the bottleneck leg limits achievable position size.
    """
    min_liq = min(liquidities) if liquidities else Decimal("0")
    return (deviation * min_liq * ARB_CAPTURE_RATE).quantize(Decimal("0.01"))


# ─────────────────────────────────────────────────────────────────────────────
# Constraint factories
# ─────────────────────────────────────────────────────────────────────────────

def complement_constraint(
    market_id: str,
    tolerance: Decimal = COMPLEMENT_TOLERANCE,
) -> Constraint:
    """
    Fires when |YES_price + NO_price − 1| > tolerance.

    Underpriced (sum < 1):  buy YES + buy NO for sum, collect $1.00 at resolution.
    Overpriced  (sum > 1):  sell YES + sell NO (requires shorting; harder in practice).
    """
    def check(markets: Dict[str, Market]) -> Optional[Violation]:
        m = markets.get(market_id)
        if m is None:
            return None

        total     = m.complement_sum
        deviation = (total - Decimal("1")).copy_abs()
        if deviation <= tolerance:
            return None

        sev = _severity(deviation, m.liquidity_usdc)

        if total < Decimal("1"):
            arb    = _arb_usdc(Decimal("1") - total, m.liquidity_usdc)
            detail = (
                f"sum = {total:.4f} < 1.00  →  "
                f"buy YES @ {m.yes_price} + buy NO @ {m.no_price}; "
                f"guaranteed $1.00 payout at resolution  (gap = {Decimal('1') - total:.4f})"
            )
        else:
            arb    = _arb_usdc(total - Decimal("1"), m.liquidity_usdc) * Decimal("0.5")
            detail = (
                f"sum = {total:.4f} > 1.00  →  "
                f"sell YES @ {m.yes_price} + sell NO @ {m.no_price}  "
                f"(shorting required; collect excess {total - Decimal('1'):.4f})"
            )

        return Violation(
            constraint_id   = f"complement:{market_id}",
            constraint_type = ConstraintType.COMPLEMENT,
            description     = f"Complement pricing — {m.question}",
            market_ids      = (market_id,),
            deviation       = deviation,
            severity        = sev,
            arb_profit_usdc = arb,
            detail          = detail,
        )

    return Constraint(
        id          = f"complement:{market_id}",
        type        = ConstraintType.COMPLEMENT,
        description = f"YES + NO ≈ 1.0  [{market_id}]",
        market_ids  = (market_id,),
        check       = check,
    )


def implication_constraint(
    antecedent_id: str,
    consequent_id: str,
    label: str = "",
) -> Constraint:
    """
    If event A implies event B (A ⊆ B), then P(B) ≥ P(A).

    antecedent = more specific event, e.g. "Fed hike in Q1 2026"
    consequent = more general event,  e.g. "Fed hike in 2026"

    Trade signal: buy the underpriced consequent (B) and fade the antecedent (A).
    """
    cid  = f"impl:{antecedent_id}→{consequent_id}"
    desc = label or f"P({consequent_id}) ≥ P({antecedent_id})"

    def check(markets: Dict[str, Market]) -> Optional[Violation]:
        a = markets.get(antecedent_id)
        b = markets.get(consequent_id)
        if a is None or b is None:
            return None
        if b.yes_price >= a.yes_price:
            return None

        deviation = a.yes_price - b.yes_price
        sev       = _severity(deviation, a.liquidity_usdc, b.liquidity_usdc)
        arb       = _arb_usdc(deviation, a.liquidity_usdc, b.liquidity_usdc)

        return Violation(
            constraint_id   = cid,
            constraint_type = ConstraintType.IMPLICATION,
            description     = desc,
            market_ids      = (antecedent_id, consequent_id),
            deviation       = deviation,
            severity        = sev,
            arb_profit_usdc = arb,
            detail          = (
                f"P(antecedent  '{a.question[:55]}') = {a.yes_price}  >  "
                f"P(consequent  '{b.question[:55]}') = {b.yes_price}  "
                f"[Δ = {deviation}]  →  buy {consequent_id} YES (underpriced by {deviation})"
            ),
        )

    return Constraint(
        id          = cid,
        type        = ConstraintType.IMPLICATION,
        description = desc,
        market_ids  = (antecedent_id, consequent_id),
        check       = check,
    )


def conjunction_constraint(
    and_id: str,
    a_id: str,
    b_id: str,
    label: str = "",
) -> Constraint:
    """
    P(A ∧ B) ≤ min(P(A), P(B)).

    A joint event cannot be more likely than either of its components.
    Trade: sell the overpriced AND market; hedge with NO positions on A or B.
    """
    cid  = f"conj:{and_id}<=min({a_id},{b_id})"
    desc = label or f"P({and_id}) ≤ min(P({a_id}), P({b_id}))"

    def check(markets: Dict[str, Market]) -> Optional[Violation]:
        m_and = markets.get(and_id)
        m_a   = markets.get(a_id)
        m_b   = markets.get(b_id)
        if None in (m_and, m_a, m_b):
            return None

        upper = min(m_a.yes_price, m_b.yes_price)
        if m_and.yes_price <= upper:
            return None

        deviation = m_and.yes_price - upper
        sev       = _severity(deviation, m_and.liquidity_usdc, m_a.liquidity_usdc, m_b.liquidity_usdc)
        arb       = _arb_usdc(deviation, m_and.liquidity_usdc)

        return Violation(
            constraint_id   = cid,
            constraint_type = ConstraintType.CONJUNCTION,
            description     = desc,
            market_ids      = (and_id, a_id, b_id),
            deviation       = deviation,
            severity        = sev,
            arb_profit_usdc = arb,
            detail          = (
                f"P('{m_and.question[:55]}') = {m_and.yes_price}  "
                f"> min(P({a_id})={m_a.yes_price}, P({b_id})={m_b.yes_price}) = {upper}  "
                f"[Δ = {deviation}]  →  sell {and_id} YES; hedge with {a_id}/{b_id} NO"
            ),
        )

    return Constraint(
        id          = cid,
        type        = ConstraintType.CONJUNCTION,
        description = desc,
        market_ids  = (and_id, a_id, b_id),
        check       = check,
    )


def disjunction_lower_constraint(
    or_id: str,
    a_id: str,
    b_id: str,
    label: str = "",
) -> Constraint:
    """
    P(A ∨ B) ≥ max(P(A), P(B)).

    An OR event must be at least as likely as the most likely of its components.
    Trade: buy the underpriced OR market.
    """
    cid  = f"disj_lo:{or_id}>=max({a_id},{b_id})"
    desc = label or f"P({or_id}) ≥ max(P({a_id}), P({b_id}))"

    def check(markets: Dict[str, Market]) -> Optional[Violation]:
        m_or = markets.get(or_id)
        m_a  = markets.get(a_id)
        m_b  = markets.get(b_id)
        if None in (m_or, m_a, m_b):
            return None

        lower = max(m_a.yes_price, m_b.yes_price)
        if m_or.yes_price >= lower:
            return None

        deviation = lower - m_or.yes_price
        sev       = _severity(deviation, m_or.liquidity_usdc, m_a.liquidity_usdc, m_b.liquidity_usdc)
        arb       = _arb_usdc(deviation, m_or.liquidity_usdc)

        return Violation(
            constraint_id   = cid,
            constraint_type = ConstraintType.DISJUNCTION_LO,
            description     = desc,
            market_ids      = (or_id, a_id, b_id),
            deviation       = deviation,
            severity        = sev,
            arb_profit_usdc = arb,
            detail          = (
                f"P('{m_or.question[:55]}') = {m_or.yes_price}  "
                f"< max(P({a_id})={m_a.yes_price}, P({b_id})={m_b.yes_price}) = {lower}  "
                f"[Δ = {deviation}]  →  buy {or_id} YES (underpriced OR event)"
            ),
        )

    return Constraint(
        id          = cid,
        type        = ConstraintType.DISJUNCTION_LO,
        description = desc,
        market_ids  = (or_id, a_id, b_id),
        check       = check,
    )


def disjunction_upper_constraint(
    or_id: str,
    a_id: str,
    b_id: str,
    label: str = "",
) -> Constraint:
    """
    P(A ∨ B) ≤ P(A) + P(B).

    An OR event cannot exceed the sum of its components (inclusion-exclusion floor).
    Trade: sell the overpriced OR market.
    """
    cid  = f"disj_hi:{or_id}<=(P{a_id}+P{b_id})"
    desc = label or f"P({or_id}) ≤ P({a_id}) + P({b_id})"

    def check(markets: Dict[str, Market]) -> Optional[Violation]:
        m_or = markets.get(or_id)
        m_a  = markets.get(a_id)
        m_b  = markets.get(b_id)
        if None in (m_or, m_a, m_b):
            return None

        upper = m_a.yes_price + m_b.yes_price
        if m_or.yes_price <= upper:
            return None

        deviation = m_or.yes_price - upper
        sev       = _severity(deviation, m_or.liquidity_usdc, m_a.liquidity_usdc, m_b.liquidity_usdc)
        arb       = _arb_usdc(deviation, m_or.liquidity_usdc)

        return Violation(
            constraint_id   = cid,
            constraint_type = ConstraintType.DISJUNCTION_HI,
            description     = desc,
            market_ids      = (or_id, a_id, b_id),
            deviation       = deviation,
            severity        = sev,
            arb_profit_usdc = arb,
            detail          = (
                f"P('{m_or.question[:55]}') = {m_or.yes_price}  "
                f"> P({a_id})={m_a.yes_price} + P({b_id})={m_b.yes_price} = {upper}  "
                f"[Δ = {deviation}]  →  sell {or_id} YES (overpriced OR event)"
            ),
        )

    return Constraint(
        id          = cid,
        type        = ConstraintType.DISJUNCTION_HI,
        description = desc,
        market_ids  = (or_id, a_id, b_id),
        check       = check,
    )


def mutual_exclusivity_constraint(
    market_ids: Tuple[str, ...],
    label: str = "",
    tolerance: Decimal = MUT_EXCL_TOLERANCE,
) -> Constraint:
    """
    Σ P(i) ≤ 1.0 + tolerance  for a set of mutually exclusive outcomes.

    Trade: sell YES on all outcomes; guaranteed to collect Σ prices but pay
    at most $1.00, pocketing the difference.
    """
    cid  = "mut_excl:" + "+".join(market_ids)
    desc = label or f"Σ P(i) ≤ 1.0  [{', '.join(market_ids)}]"

    def check(markets: Dict[str, Market]) -> Optional[Violation]:
        ms = [markets.get(mid) for mid in market_ids]
        if any(m is None for m in ms):
            return None

        total     = sum(m.yes_price for m in ms)  # type: ignore[union-attr]
        deviation = total - Decimal("1")
        if deviation <= tolerance:
            return None

        liq_vals = tuple(m.liquidity_usdc for m in ms)  # type: ignore[union-attr]
        sev      = _severity(deviation, *liq_vals)
        arb      = _arb_usdc(deviation, *liq_vals)

        terms = "  +  ".join(
            f"P({m.id})={m.yes_price}" for m in ms  # type: ignore[union-attr]
        )
        return Violation(
            constraint_id   = cid,
            constraint_type = ConstraintType.MUT_EXCL,
            description     = desc,
            market_ids      = tuple(market_ids),
            deviation       = deviation,
            severity        = sev,
            arb_profit_usdc = arb,
            detail          = (
                f"{terms}  =  {total}  >  1.0  [Δ = {deviation}]  →  "
                f"sell YES on all outcomes; collect {total} per unit, pay out $1.00"
            ),
        )

    return Constraint(
        id          = cid,
        type        = ConstraintType.MUT_EXCL,
        description = desc,
        market_ids  = tuple(market_ids),
        check       = check,
    )


def exhaustive_partition_constraint(
    market_ids: Tuple[str, ...],
    label: str = "",
    tolerance: Decimal = PARTITION_TOLERANCE,
) -> Constraint:
    """
    Σ P(i) = 1.0 ± tolerance  for an exhaustive partition of the outcome space.

    Underprice (sum < 1): buy all outcomes for Σ prices, collect $1.00 at resolution.
    Overprice  (sum > 1): sell all outcomes, collect excess over $1.00.
    """
    cid  = "partition:" + "+".join(market_ids)
    desc = label or f"Σ P(i) = 1.0  [{', '.join(market_ids)}]"

    def check(markets: Dict[str, Market]) -> Optional[Violation]:
        ms = [markets.get(mid) for mid in market_ids]
        if any(m is None for m in ms):
            return None

        total     = sum(m.yes_price for m in ms)  # type: ignore[union-attr]
        deviation = (total - Decimal("1")).copy_abs()
        if deviation <= tolerance:
            return None

        liq_vals = tuple(m.liquidity_usdc for m in ms)  # type: ignore[union-attr]
        sev      = _severity(deviation, *liq_vals)
        arb      = _arb_usdc(deviation, *liq_vals)

        if total < Decimal("1"):
            gap    = Decimal("1") - total
            detail_suffix = (
                f"=  {total}  <  1.0  [gap = {gap:.4f}]  →  "
                f"buy all outcomes for {total}; guaranteed $1.00 payout at resolution"
            )
        else:
            excess = total - Decimal("1")
            detail_suffix = (
                f"=  {total}  >  1.0  [excess = {excess:.4f}]  →  "
                f"sell all outcomes; collect {total}, pay out $1.00"
            )

        terms = "  +  ".join(
            f"P({m.id})={m.yes_price}" for m in ms  # type: ignore[union-attr]
        )
        return Violation(
            constraint_id   = cid,
            constraint_type = ConstraintType.PARTITION,
            description     = desc,
            market_ids      = tuple(market_ids),
            deviation       = deviation,
            severity        = sev,
            arb_profit_usdc = arb,
            detail          = f"{terms}  {detail_suffix}",
        )

    return Constraint(
        id          = cid,
        type        = ConstraintType.PARTITION,
        description = desc,
        market_ids  = tuple(market_ids),
        check       = check,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constraint Engine
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintEngine:
    """
    Registry of markets and constraints.  Evaluates all constraints against
    the current market snapshot and returns ranked violations.

    Designed for extensibility — connect live Polymarket data like this:

        engine = ConstraintEngine()
        engine.load_markets(polymarket_client.get_all_markets())
        engine.register_many(build_constraints())
        violations = engine.violations_by_severity()
    """

    def __init__(self) -> None:
        self._markets: Dict[str, Market]     = {}
        self._constraints: List[Constraint]  = []

    # ── Market registry ──────────────────────────────────────────────────────

    def add_market(self, market: Market) -> None:
        self._markets[market.id] = market

    def load_markets(self, markets: Iterable[Market]) -> None:
        for m in markets:
            self.add_market(m)

    def get_market(self, market_id: str) -> Optional[Market]:
        return self._markets.get(market_id)

    def market_count(self) -> int:
        return len(self._markets)

    def market_ids(self) -> List[str]:
        return list(self._markets.keys())

    # ── Constraint registry ──────────────────────────────────────────────────

    def register(self, constraint: Constraint) -> None:
        self._constraints.append(constraint)

    def register_many(self, constraints: Iterable[Constraint]) -> None:
        for c in constraints:
            self.register(c)

    def constraint_count(self) -> int:
        return len(self._constraints)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self) -> List[Violation]:
        """
        Run every registered constraint against the current market snapshot.
        A bad constraint never aborts the scan — it is skipped with a warning.
        """
        violations: List[Violation] = []
        for constraint in self._constraints:
            try:
                result = constraint.check(self._markets)
            except Exception as exc:
                print(f"[WARN] Constraint '{constraint.id}' raised: {exc}")
                continue
            if result is not None:
                violations.append(result)
        return violations

    def violations_by_severity(self) -> List[Violation]:
        """Evaluate and return violations sorted highest → lowest severity."""
        return sorted(self.evaluate(), reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reporter — console rendering (pure presentation, zero business logic)
# ─────────────────────────────────────────────────────────────────────────────

class Reporter:
    """
    Renders a ranked violation report to stdout.

    All formatting is isolated here so the engine and models can run
    headlessly in tests, pipelines, or downstream systems.
    """

    WIDTH   = 80
    HEAVY   = "═" * WIDTH
    THIN    = "─" * WIDTH
    INDENT  = "    "

    @classmethod
    def render(
        cls,
        violations: List[Violation],
        market_count: int,
        constraint_count: int,
    ) -> None:
        cls._header(market_count, constraint_count, len(violations))
        if not violations:
            print("\n  ✓  No violations detected — all prices are consistent.\n")
            print(cls.HEAVY)
            return

        print(f"\n  Ranked by severity  (highest = most exploitable)\n")
        for rank, v in enumerate(violations, start=1):
            cls._violation_block(rank, v)

        cls._summary(violations)

    @classmethod
    def _header(cls, markets: int, constraints: int, violations: int) -> None:
        print()
        print(cls.HEAVY)
        print("  POLYMARKET CROSS-MARKET CONSTRAINT ENGINE — VIOLATION SCAN")
        print(f"  Markets scanned : {markets}  |  Constraints checked : {constraints}  |  Violations : {violations}")
        print(cls.HEAVY)

    @classmethod
    def _violation_block(cls, rank: int, v: Violation) -> None:
        ids_str = ", ".join(v.market_ids)
        if len(ids_str) > 55:
            ids_str = ids_str[:52] + "…"

        print(cls.THIN)
        print(
            f"  #{rank:<3} SEVERITY {v.severity:.4f}"
            f"  │  {v.constraint_type.value.upper()}"
            f"  │  Δ = {v.deviation}"
        )
        print(f"  {cls.INDENT}Markets     » {ids_str}")
        print(f"  {cls.INDENT}Arb profit  » ${v.arb_profit_usdc:>9,.2f}  (est. @ {int(ARB_CAPTURE_RATE*100)}% capture)")
        desc = v.description if len(v.description) <= cls.WIDTH - 20 else v.description[:cls.WIDTH - 21] + "…"
        print(f"  {cls.INDENT}Rule        » {desc}")

        # Wrap detail text at a comfortable width
        wrap_width = cls.WIDTH - len(cls.INDENT) - 16
        detail_lines = textwrap.wrap(v.detail, width=wrap_width)
        for i, line in enumerate(detail_lines):
            prefix = f"  {cls.INDENT}Trade       » " if i == 0 else f"  {cls.INDENT}              "
            print(prefix + line)

    @classmethod
    def _summary(cls, violations: List[Violation]) -> None:
        counts    = Counter(v.constraint_type.value for v in violations)
        total_arb = sum(v.arb_profit_usdc for v in violations)

        print(cls.THIN)
        print()
        print("  SUMMARY BY CONSTRAINT TYPE")
        print()

        max_name_len = max(len(k) for k in counts)
        for ctype_val, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            bar = "█" * count
            print(f"  {cls.INDENT}{count:>2}x  {ctype_val:<{max_name_len}}  {bar}")

        print()
        print(f"  {'Total estimated arb opportunity':.<42} ${total_arb:>10,.2f}")
        print()
        print(cls.HEAVY)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Demo dataset — intentionally-mispriced markets
# ─────────────────────────────────────────────────────────────────────────────
# Every market below was deliberately priced to violate at least one constraint.
# Comments flag the violation type and magnitude.

def _m(
    id: str, question: str,
    yes: str, no: str,
    vol: str, liq: str,
    tags: Tuple[str, ...] = (),
) -> Market:
    return Market(
        id=id, question=question,
        yes_price=Decimal(yes), no_price=Decimal(no),
        volume_usdc=Decimal(vol), liquidity_usdc=Decimal(liq),
        tags=tags,
    )


# ── Fed rate-hike implication chain ────────────────────────────────────────
# Q1 ⊂ H1 ⊂ full-year 2026, so P(2026) ≥ P(H1) ≥ P(Q1).
# Priced in reverse: P(Q1)=0.55 > P(H1)=0.38 > P(2026)=0.30  ← VIOLATIONS

FED_Q1   = _m("fed_q1_hike",    "Fed raises rates at least once in Q1 2026",
              "0.55", "0.45", "45000", "12000", ("fed", "q1"))
FED_H1   = _m("fed_h1_hike",    "Fed raises rates at least once in H1 2026",
              "0.38", "0.62", "32000", "9500",  ("fed", "h1"))
FED_2026 = _m("fed_2026_hike",  "Fed raises rates at least once in 2026",
              "0.30", "0.70", "28000", "8000",  ("fed",))

# ── Complement pricing violations ──────────────────────────────────────────
# YES + NO = 0.88;  arb gap = $0.12 per share

SPX_6K   = _m("spx_6k",       "S&P 500 above 6,000 at year-end 2026",
              "0.51", "0.37", "210000", "55000", ("equities", "spx"))
BTC_150K = _m("btc_150k",     "Bitcoin above $150,000 by year-end 2026",
              "0.33", "0.55",  "95000", "28000", ("crypto", "btc"))

# ── NY Senate — mutual exclusivity violation ────────────────────────────────
# Only one candidate can win; probabilities sum to 1.20  ← VIOLATION

NY_ADAMS = _m("ny_adams",     "Adams wins the 2026 NY Senate race",
              "0.52", "0.48", "18000", "5500",  ("politics",))
NY_CHEN  = _m("ny_chen",      "Chen wins the 2026 NY Senate race",
              "0.38", "0.62", "15000", "4800",  ("politics",))
NY_OTHER = _m("ny_other",     "Other candidate wins the 2026 NY Senate race",
              "0.30", "0.70",  "8000", "2500",  ("politics",))
# Sum = 0.52 + 0.38 + 0.30 = 1.20  (Δ = 0.20 above the 1.0 ceiling)

# ── Macro conjunction violation ────────────────────────────────────────────
# P(A ∧ B) = 0.58  >  min(P(A)=0.45, P(B)=0.50) = 0.45  ← VIOLATION

GDP_3PCT    = _m("gdp_3pct",     "US GDP growth > 3% in 2026",
                 "0.45", "0.55", "62000", "18000", ("macro",))
UNEMP_4PCT  = _m("unemp_4pct",  "US unemployment < 4% in 2026",
                 "0.50", "0.50", "58000", "16500", ("macro",))
GDP_AND_UN  = _m("gdp_and_unemp","US GDP > 3% AND unemployment < 4% simultaneously in 2026",
                 "0.58", "0.42", "25000",  "8000", ("macro",))
# min(0.45, 0.50) = 0.45 but AND market trades at 0.58  (Δ = 0.13)

# ── Disjunction lower-bound violation ─────────────────────────────────────
# P(A ∨ B) = 0.28  <  max(P(A)=0.42, P(B)=0.35) = 0.42  ← VIOLATION

TECH_20   = _m("tech_20pct",   "Tech sector rises > 20% in 2026",
               "0.42", "0.58", "88000", "24000", ("equities",))
ENERGY_20 = _m("energy_20pct", "Energy sector rises > 20% in 2026",
               "0.35", "0.65", "44000", "13000", ("equities",))
TECH_OR_E = _m("tech_or_en",   "Tech OR Energy sector rises > 20% in 2026",
               "0.28", "0.72", "19000",  "6000", ("equities",))
# max(0.42, 0.35) = 0.42 but OR market trades at 0.28  (Δ = 0.14)

# ── Disjunction upper-bound violation ─────────────────────────────────────
# P(A ∨ B) = 0.85  >  P(A) + P(B) = 0.45 + 0.30 = 0.75  ← VIOLATION

AI_MILE  = _m("ai_milestone",  "AGI safety-benchmark milestone achieved in Q2 2026",
              "0.45", "0.55",  "71000", "20000", ("tech",))
EV_MILE  = _m("ev_milestone",  "EV battery record energy-density set in Q2 2026",
              "0.30", "0.70",  "39000", "11000", ("tech",))
AI_OR_EV = _m("ai_or_ev",      "AI or EV milestone achieved in Q2 2026",
              "0.85", "0.15",  "52000", "15000", ("tech",))
# P(AI) + P(EV) = 0.75 but OR market trades at 0.85  (Δ = 0.10)

# ── BTC price ranges — exhaustive partition violation ─────────────────────
# These three ranges cover the entire outcome space (exhaustive).
# Sum = 0.15 + 0.35 + 0.38 = 0.88  <  1.0  ← PARTITION VIOLATION

BTC_LO  = _m("btc_under100k", "BTC price below $100k at year-end 2026",
             "0.15", "0.85", "34000", "10000", ("crypto", "price"))
BTC_MID = _m("btc_100_150k",  "BTC price between $100k and $150k at year-end 2026",
             "0.35", "0.65", "51000", "15000", ("crypto", "price"))
BTC_HI  = _m("btc_over150k",  "BTC price above $150k at year-end 2026",
             "0.38", "0.62", "48000", "14000", ("crypto", "price"))
# 0.15 + 0.35 + 0.38 = 0.88;  $0.12 arb gap (buy all three, collect $1.00)


DEMO_MARKETS: Tuple[Market, ...] = (
    FED_Q1, FED_H1, FED_2026,
    SPX_6K, BTC_150K,
    NY_ADAMS, NY_CHEN, NY_OTHER,
    GDP_3PCT, UNEMP_4PCT, GDP_AND_UN,
    TECH_20, ENERGY_20, TECH_OR_E,
    AI_MILE, EV_MILE, AI_OR_EV,
    BTC_LO, BTC_MID, BTC_HI,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constraint wiring for the demo dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_constraints() -> List[Constraint]:
    """
    Returns all constraints for DEMO_MARKETS.

    To add a new constraint: call the appropriate factory with the relevant
    market IDs and append the result.  No other changes are required.
    """
    cs: List[Constraint] = []

    # Complement pricing — must hold for every single-name market
    for mid in ("spx_6k", "btc_150k"):
        cs.append(complement_constraint(mid))

    # Fed rate-hike implication chain
    # Q1 ⊂ H1: P(H1) must be ≥ P(Q1)
    cs.append(implication_constraint(
        "fed_q1_hike", "fed_h1_hike",
        label="Fed H1 rate-hike probability must exceed Q1 (Q1 ⊂ H1)",
    ))
    # H1 ⊂ 2026: P(2026) must be ≥ P(H1)
    cs.append(implication_constraint(
        "fed_h1_hike", "fed_2026_hike",
        label="Fed 2026 rate-hike probability must exceed H1 (H1 ⊂ 2026)",
    ))
    # Q1 ⊂ 2026: transitive check — often the largest violation
    cs.append(implication_constraint(
        "fed_q1_hike", "fed_2026_hike",
        label="Fed 2026 rate-hike probability must exceed Q1 (transitive: Q1 ⊂ 2026)",
    ))

    # Macro conjunction
    cs.append(conjunction_constraint(
        "gdp_and_unemp", "gdp_3pct", "unemp_4pct",
        label="P(GDP>3% AND Unemp<4%) ≤ min(P(GDP>3%), P(Unemp<4%))",
    ))

    # Sector disjunction — lower bound
    cs.append(disjunction_lower_constraint(
        "tech_or_en", "tech_20pct", "energy_20pct",
        label="P(Tech OR Energy >20%) ≥ max(P(Tech>20%), P(Energy>20%))",
    ))

    # Tech/EV disjunction — upper bound
    cs.append(disjunction_upper_constraint(
        "ai_or_ev", "ai_milestone", "ev_milestone",
        label="P(AI or EV milestone) ≤ P(AI milestone) + P(EV milestone)",
    ))

    # NY Senate — at most one candidate can win
    cs.append(mutual_exclusivity_constraint(
        ("ny_adams", "ny_chen", "ny_other"),
        label="NY Senate candidates are mutually exclusive (one winner only)",
    ))

    # BTC year-end price ranges — exactly one range must cover the outcome
    cs.append(exhaustive_partition_constraint(
        ("btc_under100k", "btc_100_150k", "btc_over150k"),
        label="BTC year-end price ranges form an exhaustive partition of outcome space",
    ))

    return cs


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    engine = ConstraintEngine()
    engine.load_markets(DEMO_MARKETS)
    engine.register_many(build_demo_constraints())

    violations = engine.violations_by_severity()

    Reporter.render(
        violations       = violations,
        market_count     = engine.market_count(),
        constraint_count = engine.constraint_count(),
    )


if __name__ == "__main__":
    main()
