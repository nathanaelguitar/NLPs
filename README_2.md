# Market Intelligence Engine — From Learning to Production

> Applying everything learned in this NLP curriculum to build a live, API-driven system that discovers comparable companies, detects emerging themes, and surfaces market signals from raw text — automatically.

---

## The Big Picture

This project graduates from static CSV datasets to a **living pipeline**: we pull fresh business descriptions, filings, and news from public APIs, run the same LDA and Doc2Vec machinery we built on the Middle 1000, and produce a real-time comparable-companies engine with interactive dashboards.

The core thesis: **every NLP technique we learned maps directly onto a production workflow.**

```
APIs (SEC / Yahoo Finance / News)
        │
        ▼
  Text Ingestion & Cleaning         ← regex, NLTK, spaCy tokenization
        │
        ▼
  Corpus Representation             ← bag-of-words, TF-IDF, co-occurrence
        │
        ├──► LDA Topic Modeling     ← tomotopy  (industry group discovery)
        │          │
        │          └──► Hellinger Similarity → Comparable Companies List
        │
        └──► Doc2Vec Embeddings     ← gensim  (semantic company vectors)
                   │
                   ├──► KMeans / PCA / t-SNE  ← scikit-learn  (cluster maps)
                   └──► Nearest-Neighbor Search → Comp List (vector space)
                              │
                              ▼
                   Interactive Dashboard        ← Plotly / Dash
```

---

## What We Learned → What We Build

| Skill Acquired | Real-World Component |
|---|---|
| LDA topic modeling on descriptions | Cluster 5,000+ companies into 30–50 themes monthly |
| Hellinger distance for comp selection | Auto-generate peer groups per ticker, updated weekly |
| Doc2Vec document embeddings | Encode new filings into the same 150-d space overnight |
| Word vector arithmetic (`genetic + oncology`) | Semantic search: "find companies like Moderna but in diagnostics" |
| PCA / t-SNE on embeddings | Interactive 2D map of the entire public equity universe |
| KMeans on Doc2Vec clusters | Identify emerging sub-sectors before GICS taxonomy catches up |
| GICS cross-tab analysis | Validate auto-generated clusters against ground-truth labels |
| pyLDAvis topic visualization | Serve a living "What is this company about?" dashboard |

---

## Data Sources & APIs

### 1. SEC EDGAR — Full-Text Search API
- **Endpoint:** `https://efts.sec.gov/LATEST/search-index?q="..."&dateRange=custom`
- **What we pull:** Business description sections from 10-K filings (Item 1)
- **Why:** Ground-truth company language, updated each reporting cycle
- **Library:** `requests` + `BeautifulSoup` for HTML stripping

```python
import requests

def fetch_10k_description(cik: str) -> str:
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    filings = requests.get(url, headers={"User-Agent": "your@email.com"}).json()
    # locate most recent 10-K accession number, fetch full filing text
    ...
```

### 2. Yahoo Finance (yfinance) — Ticker Metadata
- **What we pull:** Sector, industry, market cap, long business summary
- **Why:** Fast bootstrap for any ticker without filing lag
- **Library:** `yfinance`

```python
import yfinance as yf

info = yf.Ticker("NVDA").info
description = info["longBusinessSummary"]
sector      = info["sector"]
market_cap  = info["marketCap"]
```

### 3. NewsAPI — Real-Time News Corpus
- **Endpoint:** `https://newsapi.org/v2/everything?q={company_name}`
- **What we pull:** Article headlines + descriptions for sentiment and theme detection
- **Why:** LDA trained on news reveals emerging narratives before they hit financials
- **Library:** `requests`

### 4. Alpha Vantage — Company Overview Endpoint
- **Endpoint:** `https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}`
- **What we pull:** Description, sector, exchange, EPS, P/E — enriches the embedding metadata
- **Free tier:** 25 requests/day, sufficient for incremental updates

### 5. OpenBB Platform (optional, open-source Bloomberg alternative)
- Unified Python interface to SEC, Yahoo, FRED, and 30+ other sources
- Single `openbb.equity.fundamental.overview("AAPL")` call replaces four separate API integrations

---

## System Architecture

### Module 1 — Ingestion & Cleaning (`ingest.py`)

Pulls raw text from APIs and applies the same preprocessing pipeline we perfected on the Middle 1000:

```python
import re, nltk
from nltk.corpus import stopwords

STOPS = set(stopwords.words("english")) | {
    "company", "provides", "offers", "products", "services",  # boilerplate killers
    "inc", "corp", "ltd", "group", "holdings"
}

def clean_description(raw: str) -> str:
    text = raw.lower()
    text = re.sub(r"[^a-z\s]", " ", text)           # strip punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPS and len(t) > 2]
    return " ".join(tokens)
```

**Key lesson applied:** The GICS alignment work taught us that boilerplate language (words like "provides", "offers") destroys topic coherence. We built an extended stopword list specifically for financial text.

---

### Module 2 — LDA Topic Engine (`lda_engine.py`)

Trains (or incrementally updates) a tomotopy LDA model on the live corpus:

```python
import tomotopy as tp
import pickle

def train_or_update_lda(docs: list[str], model_path="lda_live.mdl", k=30):
    """
    If a model exists, infer topics for new docs without full retraining.
    Full retrain runs monthly on a fresh corpus snapshot.
    """
    try:
        mdl = tp.LDAModel.load(model_path)
        print(f"Loaded existing {k}-topic model")
    except FileNotFoundError:
        mdl = tp.LDAModel(k=k, min_df=5, rm_top=20)
        for doc in docs:
            mdl.add_doc(doc.split())
        mdl.train(1000)
        mdl.save(model_path)
    return mdl

def get_topic_vector(mdl, text: str) -> list[float]:
    doc = mdl.make_doc(text.split())
    topic_dist, _ = mdl.infer(doc)
    return list(topic_dist)
```

**Key lesson applied:** From the GICS cross-tab work we know that 24–30 topics captures most meaningful industry segmentation. We start at 30 and tune via coherence score.

---

### Module 3 — Doc2Vec Embedding Store (`embeddings.py`)

Keeps a live Doc2Vec model that can infer vectors for new companies without retraining:

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def infer_vector(model: Doc2Vec, text: str, steps=50) -> list[float]:
    """Infer a 150-d vector for any new company description."""
    tokens = text.split()
    return model.infer_vector(tokens, steps=steps).tolist()

def find_nearest_companies(model, query_vec, df, topn=10):
    """
    Given a query vector (could be arithmetic of multiple companies),
    return the closest companies from the stored Doc2Vec docvecs.
    """
    similar = model.dv.most_similar(positive=[query_vec], topn=topn + 1)
    tickers = [tag for tag, _ in similar]
    return df[df["ticker"].isin(tickers)][["ticker", "compustat_name", "sector"]]
```

**Key lesson applied:** The `u4m2` notebook showed that vector arithmetic (`social_media + search_engine + advertisement`) finds semantically coherent peers. We expose this as an API endpoint so analysts can compose concept queries.

---

### Module 4 — Comparable Companies API (`comps_api.py`)

A lightweight FastAPI service that wraps everything:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Comps Engine")

class CompsRequest(BaseModel):
    ticker: str
    method: str = "doc2vec"   # or "lda_hellinger"
    topn: int = 10

@app.get("/comps/{ticker}")
def get_comps(ticker: str, method: str = "doc2vec", topn: int = 10):
    """
    Return top-N comparable companies for a given ticker.
    Supports both LDA/Hellinger and Doc2Vec/cosine methods.
    """
    description = fetch_description(ticker)       # from yfinance or SEC
    clean = clean_description(description)

    if method == "lda_hellinger":
        vec = get_topic_vector(lda_model, clean)
        return hellinger_comps(vec, corpus_embeddings, topn)

    if method == "doc2vec":
        vec = infer_vector(d2v_model, clean)
        return find_nearest_companies(d2v_model, vec, company_df, topn)
```

---

### Module 5 — Interactive Dashboard (`dashboard.py`)

A Plotly Dash app serving three views:

**View 1: Universe Map**
- PCA/t-SNE 2D scatter of all companies colored by LDA dominant topic
- Click any point → see company name, description, and its top-5 comps

**View 2: Comps Explorer**
- Enter a ticker → get ranked comp list with similarity scores and business descriptions
- Toggle between LDA Hellinger and Doc2Vec methods to compare results

**View 3: Topic Monitor**
- Live word clouds per topic, updated weekly
- Track which topics are gaining/losing companies (signals sector rotation)
- Cross-tab heatmap: LDA topic × GICS group (the analysis we built in Module 3.2, now live)

```python
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(id="ticker-input", options=ticker_options),
    dcc.Graph(id="universe-map"),
    dcc.Graph(id="comps-table"),
])

@app.callback(Output("comps-table", "figure"), Input("ticker-input", "value"))
def update_comps(ticker):
    comps = get_comps(ticker, method="doc2vec", topn=10)
    return px.bar(comps, x="ticker", y="similarity", color="sector",
                  title=f"Top 10 Comps for {ticker}")
```

---

## Why Each Library Is Here

### `tomotopy`
Probabilistic topic modeling. Gives every company a **soft membership vector** across 30 industry topics — far richer than a hard GICS bucket. Hellinger distance between two such vectors tells us how economically similar two companies are.

### `gensim` (Doc2Vec + Word2Vec)
Encodes entire company descriptions into 150-dimensional vectors that preserve semantic relationships. Enables:
- Nearest-neighbor comp search
- Concept arithmetic: `"find me companies like Pfizer minus drugs plus devices"`
- Word-level similarity queries for theme detection

### `scikit-learn`
- **PCA / t-SNE:** Compress 150-d Doc2Vec vectors into 2D for visualization without losing cluster structure
- **KMeans:** Discover emerging sub-sectors that GICS hasn't named yet
- **StandardScaler:** Normalize embeddings before clustering so no dimension dominates

### `scipy`
Hellinger distance is the mathematically correct metric for comparing probability distributions (LDA topic vectors). Our GICS alignment work validated that it outperforms cosine similarity for this specific use case.

### `pandas` + `numpy`
The backbone. Cross-tabulation, `idxmax()` for hard topic assignment, vectorized distance computation over the entire 5,000-company universe.

### `plotly` (+ Dash)
Interactive charts that let analysts click, zoom, and filter. Static matplotlib charts work for learning; Plotly is what you ship to stakeholders.

### `pyLDAvis`
Provides interpretability for the LDA model itself — the relevance slider lets us tune how "exclusive" vs "inclusive" a topic's keywords are. Essential for naming topics correctly before they go into production labels.

---

## Roadmap

### Phase 1 — Data Foundation (Weeks 1–2)
- [ ] Build `ingest.py`: SEC EDGAR + yfinance ingestion for S&P 1500
- [ ] Build `clean.py`: extend stopword list, add bigram detection (e.g., `cloud_computing`)
- [ ] Pickle corpus snapshot, establish weekly refresh schedule

### Phase 2 — Model Training (Weeks 3–4)
- [ ] Train 30-topic tomotopy LDA on full 1500-company corpus
- [ ] Train Doc2Vec (vector_size=150, epochs=25) on same corpus
- [ ] Validate: reproduce GICS alignment cross-tab, confirm ≥85% Utilities / Banks coherence

### Phase 3 — Comps Engine (Weeks 5–6)
- [ ] Implement both Hellinger and Doc2Vec comp methods
- [ ] Build distance cutoff detector (elbow method) to trim non-comparable tail
- [ ] Add qualitative validation layer: flag comps where top topic doesn't match query company

### Phase 4 — API & Dashboard (Weeks 7–8)
- [ ] FastAPI service with `/comps/{ticker}` and `/topics/{ticker}` endpoints
- [ ] Plotly Dash universe map with live PCA scatter
- [ ] Topic monitor dashboard showing weekly topic drift

### Phase 5 — News Layer (Weeks 9–10)
- [ ] Pull NewsAPI headlines, run LDA inference (no retraining needed)
- [ ] Track topic salience in news vs topic salience in filings — divergence = signal
- [ ] Sentiment overlay using a pre-trained FinBERT model

---

## Project Structure (Target)

```
market-intelligence-engine/
├── ingest/
│   ├── sec_edgar.py          # 10-K Item 1 extraction
│   ├── yfinance_pull.py      # ticker metadata
│   └── newsapi_pull.py       # real-time news
├── clean/
│   └── text_cleaner.py       # regex + stopwords + bigrams
├── models/
│   ├── lda_engine.py         # tomotopy training + inference
│   └── embeddings.py         # Doc2Vec training + inference
├── analysis/
│   ├── comps.py              # Hellinger + Doc2Vec comp search
│   ├── gics_alignment.py     # cross-tab validation (from Module 3.2)
│   └── clustering.py         # KMeans + PCA/t-SNE
├── api/
│   └── comps_api.py          # FastAPI service
├── dashboard/
│   └── dashboard.py          # Plotly Dash app
├── data/
│   ├── corpus_snapshot.p     # weekly pickling
│   └── models/               # saved .mdl and .p files
├── requirements.txt
└── README_2.md               # this file
```

---

## Getting Started

```bash
# 1. Clone and create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add API keys to .env (never commit these)
echo "NEWSAPI_KEY=your_key_here" >> .env
echo "ALPHAVANTAGE_KEY=your_key_here" >> .env

# 3. Run initial ingestion (builds corpus_snapshot.p)
python ingest/yfinance_pull.py --universe sp1500

# 4. Train models
python models/lda_engine.py --train --k 30
python models/embeddings.py --train

# 5. Validate against GICS
python analysis/gics_alignment.py

# 6. Launch API
uvicorn api.comps_api:app --reload

# 7. Launch dashboard
python dashboard/dashboard.py
```

---

## Key Design Decisions Informed by This Curriculum

**Why two comp methods (LDA Hellinger + Doc2Vec)?**
From Module 4.1/4.2 work: LDA is better at capturing macro industry themes (Energy vs. Pharma), while Doc2Vec is better at fine-grained operational similarity within a sector. Running both and comparing gives an analyst more signal than either alone.

**Why not just use a pre-trained BERT model?**
BERT is powerful but opaque. The value of LDA in a business context is **interpretability**: you can tell a portfolio manager exactly which topics drove a similarity score. That's not noise — it's the product.

**Why Hellinger over cosine for LDA vectors?**
LDA output is a probability distribution (sums to 1). Hellinger distance is the mathematically appropriate metric for comparing distributions. Cosine similarity ignores the probabilistic structure.

**Why store weekly corpus snapshots?**
Retraining LDA from scratch is expensive. Snapshots let us roll back, diff topic drift over time, and run the model on historical slices — useful for backtesting whether the topic structure was predictive of sector moves.

---

## Prediction Market APIs — Optional Extension

> **Scope fit:** Prediction markets publish thousands of short, opinionated text questions ("Will the Fed raise rates before June?", "Which pharma company announces FDA approval first?"). That text is a corpus. The same LDA and Doc2Vec machinery we built maps directly onto it — cluster markets by theme, track which topic clusters move together, build a semantic search engine over open questions.

This section documents how to hit both major APIs correctly. They are fundamentally different systems; treating them the same is the most common mistake.

---

### Comparison at a Glance

| | Kalshi | Polymarket |
|---|---|---|
| Auth for public data | API key + RSA signature on every request | None (open REST) |
| Auth for trading | Same signed requests | Crypto wallet + Polygon tx signing |
| Data access | Private + public markets | Mostly public |
| Trading mechanism | REST calls (centralized exchange) | Blockchain transactions (on-chain settlement) |
| Infrastructure complexity | Medium | High |
| US legal status | Yes (CFTC-regulated) | No (US users restricted) |
| Best use for this project | Signed data pulls, trading signals | Free public market discovery |

---

### 1. Kalshi API

**What you get:** Full REST + WebSocket access to market data, order books, positions, and trading. CFTC-regulated, US-legal.

**The critical detail most people miss:** Kalshi uses **RSA-signed requests**, not simple bearer token auth. Every request requires a timestamp and a base64-encoded RSA signature computed from your private key. You cannot just pass an API key header and call it done.

#### Setup
1. Create account at [kalshi.com](https://kalshi.com)
2. Generate API key + RSA private key in account settings
3. Store private key as `KALSHI_PRIVATE_KEY` in `.env` — **never commit this**

#### Correct signing pattern

```python
import base64
import time
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

def kalshi_headers(method: str, path: str, private_key_pem: str, api_key: str) -> dict:
    """
    Build correctly signed headers for every Kalshi request.
    method: "GET", "POST", etc.
    path:   "/trade-api/v2/markets" (no query string)
    """
    timestamp_ms = str(int(time.time() * 1000))
    msg = (timestamp_ms + method.upper() + path).encode("utf-8")

    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None
    )
    signature = private_key.sign(msg, padding.PKCS1v15(), hashes.SHA256())
    sig_b64 = base64.b64encode(signature).decode("utf-8")

    return {
        "KALSHI-ACCESS-KEY":       api_key,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
        "Content-Type":            "application/json",
    }


def kalshi_get(path: str, params: dict = None) -> dict:
    BASE = "https://trading-api.kalshi.com"
    headers = kalshi_headers("GET", path, KALSHI_PRIVATE_KEY, KALSHI_API_KEY)
    r = requests.get(BASE + path, headers=headers, params=params)
    r.raise_for_status()
    return r.json()


# Pull all active markets
markets = kalshi_get("/trade-api/v2/markets", params={"status": "open", "limit": 200})
```

#### Useful endpoints

| Endpoint | What it returns |
|---|---|
| `GET /trade-api/v2/markets` | All markets (paginated) — title, volume, close date |
| `GET /trade-api/v2/markets/{ticker}` | Single market detail |
| `GET /trade-api/v2/markets/{ticker}/orderbook` | Live order book |
| `GET /trade-api/v2/portfolio/positions` | Your open positions |
| `POST /trade-api/v2/portfolio/orders` | Place an order |

#### Shortcut
Use Kalshi's official Python SDK (`pip install kalshi`) which handles signing for you. Only hand-roll signing if you need non-Python requests or want to understand the auth flow.

---

### 2. Polymarket API

**What you get:** Public market data at no cost. Trading is on-chain (Polygon network) and requires a crypto wallet, which is a separate integration entirely.

**The three layers — don't confuse them:**

| Layer | API | Purpose |
|---|---|---|
| Market discovery | Gamma API | List markets, search, get metadata |
| Trading | CLOB API | Order book, order placement (requires wallet) |
| User data | Data API | Positions, P&L |

#### Public market data (no auth needed)

```python
import requests

GAMMA = "https://gamma-api.polymarket.com"

def get_polymarket_markets(limit: int = 100, active_only: bool = True) -> list[dict]:
    """Fetch open Polymarket markets — no API key required."""
    params = {"limit": limit}
    if active_only:
        params["active"] = "true"
    r = requests.get(f"{GAMMA}/markets", params=params)
    r.raise_for_status()
    return r.json()

# Each market has: question, description, outcomes, volume, end_date, tags
markets = get_polymarket_markets(limit=200)
questions = [m["question"] for m in markets]
```

#### CLOB market data (for prices/order books, no auth needed for reads)

```python
CLOB = "https://clob.polymarket.com"

def get_market_price(condition_id: str) -> dict:
    """Get mid-market prices for a given market condition ID."""
    r = requests.get(f"{CLOB}/midpoint", params={"token_id": condition_id})
    r.raise_for_status()
    return r.json()
```

#### Trading (requires wallet — scope boundary)
Placing orders means signing Ethereum-style messages and submitting them to the CLOB API with a `L1_AUTH` or `L2_AUTH` scheme. This goes beyond the NLP scope of this project. If your team wants trading execution, use the official `py-clob-client` SDK — it abstracts the wallet signing.

```bash
pip install py-clob-client
```

---

### 3. Where This Fits the NLP Pipeline

Prediction market text integrates at two points in our existing architecture:

#### Point A — Corpus Extension (LDA)
Add market questions and descriptions as additional documents alongside company filings. Topics that emerge may span both ("Fed rate policy" topic will appear in both bank filings and rate-related market questions). This surfaces cross-domain thematic linkages.

```python
# In ingest/prediction_markets.py
def fetch_kalshi_corpus() -> list[str]:
    """Return cleaned text from all open Kalshi market titles + descriptions."""
    markets = kalshi_get("/trade-api/v2/markets", params={"status": "open", "limit": 1000})
    texts = []
    for m in markets.get("markets", []):
        raw = f"{m.get('title', '')} {m.get('subtitle', '')}"
        texts.append(clean_description(raw))
    return texts

def fetch_polymarket_corpus() -> list[str]:
    markets = get_polymarket_markets(limit=500)
    return [clean_description(m.get("question", "") + " " + m.get("description", ""))
            for m in markets]
```

#### Point B — Signal Detection (Doc2Vec)
Infer a Doc2Vec vector for each open market question. Find which company description vectors sit closest to a given market's vector. If a market question about "semiconductor export controls" has high similarity to a cluster of chip companies in our embedding space, those companies are the most exposed to that market's outcome.

```python
def markets_near_company(ticker: str, market_vectors: dict, company_df, d2v_model, topn=5):
    """
    For a given company, find the prediction market questions most
    semantically similar to its business description.
    """
    company_text = fetch_description(ticker)
    company_vec  = infer_vector(d2v_model, clean_description(company_text))

    scores = []
    for market_id, mvec in market_vectors.items():
        sim = cosine_similarity([company_vec], [mvec])[0][0]
        scores.append((market_id, sim))

    scores.sort(key=lambda x: -x[1])
    return scores[:topn]
```

---

### 4. What to Actually Build (Recommendation)

**If your goal is NLP signals (fits this project's scope):**
- Pull public data from both APIs — Polymarket needs no auth, Kalshi needs the signing layer
- Feed market questions into LDA as a second corpus alongside filings
- Build a "market exposure" view in the dashboard: given a company, which open prediction markets are most semantically related?

**If your goal is a trading bot (out of NLP scope, different project):**
- Kalshi: API is clean and US-legal — use their SDK, do it properly
- Polymarket: use `py-clob-client` SDK, requires wallet setup and understanding of Polygon/USDC mechanics
- Do not try to build two integrations from scratch simultaneously — the auth systems are completely different

**Don't try to unify them** via a wrapper API (e.g., OddsPapi) unless you only need normalized historical data. Those wrappers add latency and abstraction that matters if you ever want live order book access.

---

### 5. Environment Setup for Prediction Markets

```bash
# Add to requirements.txt
cryptography          # RSA signing for Kalshi
py-clob-client        # Polymarket trading (optional)

# Add to .env (never commit)
KALSHI_API_KEY=your_key_here
KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n..."
# Polymarket public data needs no keys
```

Add to `.gitignore` now:
```
.env
*.pem
*.key
```
