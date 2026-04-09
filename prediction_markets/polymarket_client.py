"""
Polymarket API client — public market data only.

Polymarket has three separate APIs. This file covers the two that need no auth:
  - Gamma API:  market discovery, metadata, search
  - CLOB API:   order book prices (read-only)

Trading on Polymarket requires a Polygon-network crypto wallet and on-chain
transaction signing. That's a separate integration — use py-clob-client if
you go that route. US users are geofenced from trading; public data is fine.

Reference:
  Gamma: https://gamma-api.polymarket.com
  CLOB:  https://clob.polymarket.com
"""

import json
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE  = "https://clob.polymarket.com"


def _parse_token_ids(raw) -> list[str]:
    """Gamma API returns clobTokenIds as a JSON-encoded string, not a list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    return []


class PolymarketClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/json"

    # ------------------------------------------------------------------
    # Gamma API — market discovery (no auth)
    # ------------------------------------------------------------------

    def get_markets(
        self,
        limit: int = 100,
        active_only: bool = True,
        closed: bool = False,
        tag: str = None,
    ) -> list[dict]:
        """
        Fetch markets from the Gamma API.

        Each market dict includes:
          question, description, outcomes, volume, liquidity,
          end_date_iso, tags, condition_id (needed for CLOB price lookups)
        """
        params: dict = {"limit": limit}
        if active_only:
            params["active"] = "true"
        if closed:
            params["closed"] = "true"
        if tag:
            params["tag"] = tag
        r = self.session.get(f"{GAMMA_BASE}/markets", params=params)
        r.raise_for_status()
        return r.json()

    def get_all_active_markets(self, page_size: int = 100) -> list[dict]:
        """Paginate through all active markets."""
        markets, offset = [], 0
        while True:
            params = {"limit": page_size, "offset": offset, "active": "true"}
            r = self.session.get(f"{GAMMA_BASE}/markets", params=params)
            r.raise_for_status()
            page = r.json()
            if not page:
                break
            markets.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return markets

    def get_market(self, condition_id: str) -> dict:
        """Single market by condition_id."""
        r = self.session.get(f"{GAMMA_BASE}/markets/{condition_id}")
        r.raise_for_status()
        return r.json()

    def search_markets(self, query: str, limit: int = 50) -> list[dict]:
        """
        Full-text search over market questions.
        Useful for finding markets semantically related to a company or topic.
        """
        params = {"q": query, "limit": limit, "active": "true"}
        r = self.session.get(f"{GAMMA_BASE}/markets", params=params)
        r.raise_for_status()
        return r.json()

    def get_events(self, limit: int = 50, active_only: bool = True) -> list[dict]:
        """
        Events group related markets (e.g. 'Fed Rate Decision' event contains
        multiple outcome markets). Useful for thematic clustering.
        """
        params = {"limit": limit}
        if active_only:
            params["active"] = "true"
        r = self.session.get(f"{GAMMA_BASE}/events", params=params)
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # CLOB API — prices / order books (no auth for reads)
    # ------------------------------------------------------------------

    def get_midpoint(self, token_id: str) -> float | None:
        """
        Get the mid-market price (0.0–1.0) for a single outcome token.
        token_id comes from market["clobTokenIds"][0] (Yes) or [1] (No).
        Returns None if no liquidity.
        """
        r = self.session.get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        mid = data.get("mid")
        return float(mid) if mid is not None else None

    def get_spread(self, token_id: str) -> dict | None:
        """Best bid/ask for a token. Returns {'bid': float, 'ask': float} or None."""
        r = self.session.get(f"{CLOB_BASE}/spread", params={"token_id": token_id})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        bid = data.get("bid")
        ask = data.get("ask")
        if bid is None or ask is None:
            return None
        return {"bid": float(bid), "ask": float(ask)}

    def get_orderbook(self, token_id: str) -> dict:
        """Full order book for a token."""
        r = self.session.get(f"{CLOB_BASE}/book", params={"token_id": token_id})
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Convenience helpers for NLP pipeline
    # ------------------------------------------------------------------

    def get_token_ids(self, market: dict) -> list[str]:
        """
        Extract CLOB token IDs from a Gamma market dict.
        The API returns clobTokenIds as a JSON-encoded string; this normalises it.
        Index 0 = Yes outcome, index 1 = No outcome.
        """
        return _parse_token_ids(market.get("clobTokenIds", []))

    def get_market_texts(self, limit: int = 500) -> list[dict]:
        """
        Return a list of dicts with 'id', 'question', 'description', 'tags'
        ready to feed into the LDA / Doc2Vec pipeline.
        """
        markets = self.get_all_active_markets(page_size=100)[:limit]
        out = []
        for m in markets:
            token_ids = _parse_token_ids(m.get("clobTokenIds", []))
            out.append({
                "id":           m.get("id", ""),
                "condition_id": m.get("conditionId", ""),
                "question":     m.get("question", ""),
                "description":  m.get("description", ""),
                "tags":         [t.get("label", "") for t in m.get("tags", [])],
                "volume":       float(m.get("volume", 0) or 0),
                "end_date":     m.get("endDateIso", ""),
                "token_ids":    token_ids,
            })
        return out
