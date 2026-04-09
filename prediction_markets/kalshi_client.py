"""
Kalshi API client with RSA request signing.

Kalshi is CFTC-regulated and US-legal. Every request (including GETs) must
be signed with your RSA private key — passing just the API key is not enough.

Setup:
  1. Create account at kalshi.com
  2. Generate API key + RSA key pair in Account > API Settings
  3. Set KALSHI_API_KEY and KALSHI_PRIVATE_KEY in your .env file

Reference: https://trading-api.kalshi.com/trade-api/v2/swagger
"""

import base64
import json
import os
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.padding import PSS, MGF1

BASE_URL      = "https://api.elections.kalshi.com"
DEMO_BASE_URL = "https://api.elections.kalshi.com"  # same endpoint; balance starts at 0 for new accounts


class KalshiClient:
    def __init__(
        self,
        api_key: str | None = None,
        private_key_pem: str | None = None,
        demo: bool = True,
    ):
        """
        Args:
            api_key:         From KALSHI_API_KEY env var if not passed directly.
            private_key_pem: PEM string from KALSHI_PRIVATE_KEY env var. Include
                             the full -----BEGIN ... END----- block.
            demo:            True = paper trading (demo-api.kalshi.co).
                             False = live trading (trading-api.kalshi.com).
        """
        self.api_key = api_key or os.environ["KALSHI_API_KEY"]

        if private_key_pem:
            pem_bytes = private_key_pem.replace("\\n", "\n").encode()
        elif "KALSHI_PRIVATE_KEY" in os.environ:
            pem_bytes = os.environ["KALSHI_PRIVATE_KEY"].replace("\\n", "\n").encode()
        else:
            # Fall back to key file path
            key_file = os.environ.get("KALSHI_KEY_FILE", "kalshi_private.pem")
            with open(key_file) as f:
                pem_bytes = f.read().encode()

        self.private_key = serialization.load_pem_private_key(pem_bytes, password=None)
        self.base    = DEMO_BASE_URL if demo else BASE_URL
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _sign(self, method: str, path: str) -> dict:
        """Return signed headers for a single request."""
        ts  = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode("utf-8")
        sig = self.private_key.sign(
            msg,
            PSS(mgf=MGF1(hashes.SHA256()), salt_length=PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY":       self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type":            "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.session.get(
            self.base + path,
            headers=self._sign("GET", path),
            params=params,
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = self.session.post(
            self.base + path,
            headers=self._sign("POST", path),
            data=json.dumps(body),
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_markets(
        self,
        status: str = "open",
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict:
        """
        Fetch a page of markets.
        status: "open" | "closed" | "settled"
        Returns dict with keys: markets, cursor (for pagination).
        """
        params: dict = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/trade-api/v2/markets", params=params)

    def get_all_open_markets(self) -> list[dict]:
        """Paginate through all open markets and return the full list."""
        markets: list[dict] = []
        cursor: str | None  = None
        while True:
            resp   = self.get_markets(status="open", limit=200, cursor=cursor)
            markets.extend(resp.get("markets", []))
            cursor = resp.get("cursor") or None
            if not cursor:
                break
        return markets

    def get_market(self, ticker: str) -> dict:
        """Single market detail by ticker (e.g. 'FED-25-D-0250')."""
        return self._get(f"/trade-api/v2/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Live order book for a market."""
        return self._get(
            f"/trade-api/v2/markets/{ticker}/orderbook",
            params={"depth": depth},
        )

    # ------------------------------------------------------------------
    # Portfolio (paper trading)
    # ------------------------------------------------------------------

    def get_balance(self) -> dict:
        return self._get("/trade-api/v2/portfolio/balance")

    def get_positions(self) -> list[dict]:
        resp = self._get("/trade-api/v2/portfolio/positions")
        return resp.get("market_positions", [])

    def get_orders(self, status: str = "resting") -> list[dict]:
        resp = self._get("/trade-api/v2/portfolio/orders", params={"status": status})
        return resp.get("orders", [])

    # ------------------------------------------------------------------
    # Trading (paper mode when demo=True)
    # ------------------------------------------------------------------

    def place_order(
        self,
        ticker:     str,
        side:       str,         # "yes" or "no"
        action:     str,         # "buy" or "sell"
        count:      int,         # number of contracts
        order_type: str,         # "limit" or "market"
        limit_price: int | None = None,  # cents (1–99), required for limit orders
    ) -> dict:
        """
        Place an order. In demo mode this is pure paper trading.

        limit_price is in cents (50 = $0.50 = 50% implied probability).
        count is number of contracts ($0.01 per cent of risk per contract).
        """
        body: dict = {
            "ticker": ticker,
            "side":   side,
            "action": action,
            "count":  count,
            "type":   order_type,
        }
        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            body["yes_price"] = limit_price if side == "yes" else (100 - limit_price)
        return self._post("/trade-api/v2/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        return self._post(f"/trade-api/v2/portfolio/orders/{order_id}/cancel", {})
