# =============================================================================
# Project - AI Investment Robot - Using LLMs for Investment Analytics
# Open-Source LLM Version: Llama 3.2 via Ollama (local, free, private)
# =============================================================================

# ── Python standard library ───────────────────────────────────────────────────
import os          # access to operating system environment variables
import json        # JSON serialization / deserialization
import asyncio     # async execution (multiple LLM calls running in parallel)
import logging     # event logging to file and terminal
import httpx       # modern async-capable HTTP client (replaces requests)
import numpy as np # vectorized numeric operations (mean, sign, arange…)
import pandas as pd # table (DataFrame) manipulation for prices and news

# ── Third-party libraries ─────────────────────────────────────────────────────
import yfinance as yf              # downloads historical stock data from Yahoo Finance

from datetime import datetime      # date and timestamp manipulation
from pathlib import Path           # cross-platform directory creation
from scipy import stats            # linear regression to compute price trend
from newsapi import NewsApiClient  # official NewsAPI SDK for news retrieval
from dotenv import load_dotenv     # loads variables from .env file into os.environ

# ── Credential loading ────────────────────────────────────────────────────────
# load_dotenv reads the .env file at the project root and injects variables
# into the process environment, making them accessible via os.getenv()
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Fail immediately with a clear message if the required key is missing.
# Prevents cryptic errors later when the API is called for the first time.
if not NEWSAPI_KEY:
    raise EnvironmentError(
        "\n[ERROR] NEWSAPI_KEY not found.\n"
        "Create a .env file based on .env.example:\n"
        "  NEWSAPI_KEY=your-newsapi-key-here\n"
    )

# Initialize the NewsAPI client with the key loaded from .env
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

# ── Configuration constants ───────────────────────────────────────────────────
# Centralizing here avoids magic numbers scattered through the code
# and makes tuning easy without touching any function logic.

LLM_MODEL        = "llama3.2"                    # Ollama model to use
OLLAMA_URL       = "http://localhost:11434/api/chat"  # REST endpoint of the local Ollama server
MAX_CONCURRENT   = 3          # max simultaneous requests to Llama (semaphore)
NEWS_PAGE_SIZE   = 20         # number of articles to fetch from NewsAPI per analysis
HISTORY_PERIOD   = "1mo"      # price history window in yfinance (1 month ≈ 30 trading days)
BENCHMARK_TICKER = "^GSPC"    # S&P 500 ticker used as market benchmark
BUY_THRESHOLD    = 0.25       # composite score above this value → BUY decision
SELL_THRESHOLD   = -0.25      # composite score below this value → SELL decision
OLLAMA_TIMEOUT   = 120        # per-call timeout in seconds for Llama requests

# ── Output directory creation ─────────────────────────────────────────────────
# exist_ok=True prevents an error if the directories already exist from prior runs
Path("outputs").mkdir(exist_ok=True)  # HTML, JSON and CSV reports
Path("logs").mkdir(exist_ok=True)     # persistent log file

# ── Logging setup ─────────────────────────────────────────────────────────────
# Dual output: writes to a file (for future auditing) AND prints to the terminal.
# FileHandler uses append mode — the log accumulates across runs, never overwrites.
logging.basicConfig(
    level=logging.INFO,                                      # minimum level: INFO (DEBUG is ignored)
    format="%(asctime)s | %(levelname)-7s | %(message)s",   # format: date | level | message
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/investment_robot.log", encoding="utf-8"),  # file handler
        logging.StreamHandler(),                                               # console handler
    ],
)
log = logging.getLogger(__name__)  # logger named after the current module


# ── FUNCTION: dsa_check_ollama ────────────────────────────────────────────────
# Checks whether the Ollama server is running locally and whether the model
# configured in LLM_MODEL is available for use.
# Aborts execution with SystemExit if either condition is not met.
def dsa_check_ollama() -> None:
    try:
        # GET /api/tags returns the list of models already downloaded in Ollama
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]

        # Uses substring match to catch tag variants such as "llama3.2:latest"
        if not any(LLM_MODEL in m for m in models):
            print(f"\n[!] Model '{LLM_MODEL}' not found in Ollama.")
            print(f"    Run: ollama pull {LLM_MODEL}")
            raise SystemExit(1)
        log.info(f"Ollama OK | model={LLM_MODEL} available")
    except httpx.ConnectError:
        # ConnectError is raised when the Ollama server is not running
        print("\n[ERROR] Cannot connect to Ollama at http://localhost:11434")
        print("  Install: https://ollama.com/download | Start: ollama serve\n")
        raise SystemExit(1)


# ── FUNCTION: dsa_interactive_menu ───────────────────────────────────────────
# Displays the welcome menu and collects from the user:
#   1. Stock ticker (validated live via yfinance)
#   2. Company name for news search (auto-detected or entered manually)
# Returns a validated (ticker, company) tuple.
def dsa_interactive_menu() -> tuple:
    print("""
+======================================================+
|          AI Investment Robot                         |
|     LLM : Llama 3.2 (Ollama local)                   |
+======================================================+

Supported companies (NYSE/NASDAQ). Examples:
  AAPL -- Apple         MSFT -- Microsoft
  GOOGL -- Alphabet     TSLA -- Tesla
  AMZN -- Amazon        NVDA -- NVIDIA
  META -- Meta          NFLX -- Netflix
""")
    while True:
        ticker = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()

        # Basic validation: ticker must be between 1 and 5 characters
        if not ticker or len(ticker) > 5:
            print("[!] Ticker must be 1-5 uppercase letters.\n")
            continue

        # Validates the ticker by querying the current price via yfinance.
        # fast_info is faster than .info because it only loads basic metadata.
        print(f"[~] Validating ticker '{ticker}' on yfinance...")
        try:
            info = yf.Ticker(ticker).fast_info
            last_price = getattr(info, "last_price", None)
            if last_price is None or last_price == 0:
                raise ValueError()
            print(f"[OK] Ticker '{ticker}' found. Last price: ${last_price:.2f}\n")
            break
        except Exception:
            print(f"[!] Ticker '{ticker}' not found.\n")

    # Tries to auto-detect the company name via yfinance
    # to save the user from typing it manually
    detected_name = ""
    try:
        print(f"[~] Fetching company info for '{ticker}'...")
        stock_info = yf.Ticker(ticker).info
        # longName takes priority over shortName (more descriptive)
        raw_name   = stock_info.get("longName") or stock_info.get("shortName") or ""

        # Strips common legal suffixes (Inc., Corp., etc.) that would pollute
        # the NewsAPI search query with irrelevant terms
        for suffix in [" Inc.", " Inc", " Corp.", " Corp", " Corporation",
                       " Ltd.", " Ltd", " LLC", " PLC", " N.V.", " S.A.", " Co."]:
            raw_name = raw_name.replace(suffix, "")
        detected_name = raw_name.strip()
    except Exception:
        pass  # silent failure — user will type the name manually

    if detected_name:
        print(f"[OK] Company detected: {detected_name}")
        # Lets the user override the auto-detected name
        override = input(f"Press Enter to use '{detected_name}' or type a different name: ").strip()
        company  = override if override else detected_name
    else:
        # Fallback: name entered manually when auto-detection fails
        while True:
            company = input("Enter the company name for news search (e.g., Apple): ").strip()
            if len(company) >= 2:
                break
            print("[!] Company name must be at least 2 characters.\n")

    return ticker, company


# ── FUNCTION: dsa_collect_news ────────────────────────────────────────────────
# Queries the NewsAPI for recent articles about the company.
# Filters out invalid articles (missing title or content removed as "[Removed]").
# Returns a list of dicts containing title, summary, URL, date and source.
def dsa_collect_news(company_name: str, page_size: int = NEWS_PAGE_SIZE) -> list:
    log.info(f"Collecting news | company={company_name}")
    try:
        # get_everything searches across all available sources
        # language='en' ensures English articles (better coverage for NYSE/NASDAQ)
        # sort_by='relevancy' prioritizes articles most related to the company
        response = newsapi.get_everything(
            q=company_name, language="en", sort_by="relevancy", page_size=page_size
        )
    except Exception as e:
        log.error(f"NewsAPI request failed | {e}")
        raise

    articles = []
    for art in response.get("articles", []):
        title   = art.get("title", "") or ""
        summary = art.get("description", "") or ""

        # NewsAPI marks articles removed by DMCA/privacy with "[Removed]"
        # These are filtered out as they contain no analyzable content
        if not title or "[Removed]" in title:
            continue

        articles.append({
            "title":        title,
            "summary":      summary,
            "url":          art.get("url", ""),
            "published_at": art.get("publishedAt", ""),
            "source":       art.get("source", {}).get("name", ""),
        })

    log.info(f"News collected | count={len(articles)}")
    if not articles:
        log.warning(f"No valid articles found for '{company_name}'")
    return articles


# ── FUNCTION: dsa_analyze_single_article ─────────────────────────────────────
# Async function that sends ONE article to Llama for sentiment analysis.
# Uses the Semaphore to cap simultaneous calls and avoid overloading the model.
# Returns the original article dict enriched with: category, score, justification.
async def dsa_analyze_single_article(article: dict, sem: asyncio.Semaphore, company_name: str, client: httpx.AsyncClient) -> dict:
    # System prompt: defines the model's role and the expected response format.
    # Strict "category - justification" format enables deterministic parsing.
    # temperature=0.1 reduces creativity and improves response consistency.
    system_prompt = (
        "You are a financial analyst specialized in market news sentiment analysis.\n"
        "Analyze the title and summary of the provided news article about " + company_name + ".\n"
        "Respond ONLY in this exact format: <category> - <short justification>\n"
        "Where <category> is exactly one of: positive, negative, neutral\n"
        "Example: negative - regulatory scandal involving user data\n"
        "Do not add anything else."
    )
    # User prompt: the article content to be classified
    user_prompt = "Title: " + article["title"] + "\nSummary: " + article["summary"]

    # Payload in the format expected by the Ollama chat API
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "stream": False,           # full response at once (no streaming)
        "options": {"temperature": 0.1},  # low temperature = more deterministic responses
    }

    # The Semaphore ensures at most MAX_CONCURRENT requests are active
    # at any time, preventing timeout and overloading the local Llama process
    async with sem:
        try:
            resp = await client.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()  # raises an exception for HTTP 4xx/5xx status codes
            raw    = resp.json()["message"]["content"].strip()
            parsed = dsa_parse_sentiment(raw)  # converts free text into a structured dict
        except Exception as e:
            # Graceful fallback: classify as neutral and keep the pipeline running
            # without interrupting the analysis of the remaining articles
            short_title = article["title"][:50]
            log.error(f"LLM call failed | title='{short_title}' | {e}")
            parsed = {"category": "neutral", "score": 0, "justification": "LLM call failed"}

    # Merges the original article fields with the returned sentiment fields
    return {**article, **parsed}


# ── FUNCTION: dsa_parse_sentiment ────────────────────────────────────────────
# Converts the LLM's free-text response into a structured dict.
# Expects the format: "positive - strong quarterly earnings beat"
# Returns: {"category": "positive", "score": 1, "justification": "..."}
def dsa_parse_sentiment(response: str) -> dict:
    # Maps text category to the numeric score used in the calculation
    VALID         = {"positive": 1, "negative": -1, "neutral": 0}
    parts         = response.strip().lower().split(" - ", 1)  # splits on first " - " occurrence
    category      = parts[0].strip()
    justification = parts[1].strip() if len(parts) > 1 else "no justification provided"

    if category not in VALID:
        # Substring search: handles cases where the LLM adds extra text before the category
        # e.g. "the sentiment is positive" → matches "positive" by substring
        for cat in VALID:
            if cat in category:
                category = cat
                break
        else:
            # Last resort: classify as neutral if no match is possible
            category      = "neutral"
            justification = "unparseable response: " + response[:100]

    return {"category": category, "score": VALID[category], "justification": justification}


# ── FUNCTION: dsa_analyze_sentiment_async ────────────────────────────────────
# Orchestrates sentiment analysis for ALL articles in parallel.
# asyncio.gather runs all coroutines simultaneously (bounded by the Semaphore).
# This reduces analysis time from ~90 s (sequential) to ~40 s (parallel, MAX_CONCURRENT=3).
async def dsa_analyze_sentiment_async(articles: list, company_name: str) -> list:
    sem = asyncio.Semaphore(MAX_CONCURRENT)  # controls maximum concurrency
    async with httpx.AsyncClient() as client:
        # Creates one async task per article
        tasks   = [dsa_analyze_single_article(a, sem, company_name, client) for a in articles]
        # gather runs all tasks in parallel and waits for all of them to finish
        results = await asyncio.gather(*tasks)

    # Tallies the sentiment distribution for the log
    pos = sum(1 for r in results if r["category"] == "positive")
    neu = sum(1 for r in results if r["category"] == "neutral")
    neg = sum(1 for r in results if r["category"] == "negative")
    log.info(f"Sentiments analyzed | positive={pos} | neutral={neu} | negative={neg}")
    return list(results)


# ── FUNCTION: dsa_collect_stock_data ─────────────────────────────────────────
# Downloads the last 30 days of prices and volumes for:
#   - The analyzed ticker (company stock)
#   - The S&P 500 benchmark (^GSPC) for alpha calculation
# Returns a dict with two DataFrames: "stock" and "sp500"
def dsa_collect_stock_data(ticker: str) -> dict:
    log.info(f"Collecting stock data | ticker={ticker}")
    data       = yf.Ticker(ticker).history(period=HISTORY_PERIOD)        # stock price history
    data_sp500 = yf.Ticker(BENCHMARK_TICKER).history(period=HISTORY_PERIOD)  # S&P 500 history
    if data.empty:
        raise ValueError(f"Ticker '{ticker}' returned no historical data.")
    log.info(f"Stock data collected | rows={len(data)}")
    return {"stock": data, "sp500": data_sp500}


# ── FUNCTION: dsa_calculate_trend ────────────────────────────────────────────
# Computes the 30-day price trend using ordinary least squares linear regression.
# Regression is more robust than comparing first vs. last price because it
# considers all data points and filters out day-to-day volatility noise.
# Returns slope (USD/day), slope_pct (%/day), R² and direction ("upward"/"downward").
def dsa_calculate_trend(close_prices: pd.Series) -> dict:
    x = np.arange(len(close_prices))   # X-axis: day index [0, 1, 2, ..., N-1]
    y = close_prices.values             # Y-axis: closing prices

    # linregress returns: slope, intercept, r_value, p_value, std_err
    slope, _, r_value, _, _ = stats.linregress(x, y)

    # Normalizes the slope to % of the mean price, allowing comparison between
    # assets with very different absolute price levels (e.g. NVDA vs AAPL)
    slope_pct = slope / close_prices.mean() * 100

    return {
        "slope":             float(slope),                      # slope in USD per day
        "slope_pct_per_day": round(float(slope_pct), 4),       # slope as % of mean price per day
        "r_squared":         round(float(r_value) ** 2, 4),    # goodness of fit (0=noise, 1=perfect)
        "trend":             "upward" if slope > 0 else "downward",
    }


# ── FUNCTION: dsa_analyze_volume ─────────────────────────────────────────────
# Compares the most recent day's volume against the 30-day average.
# High volume (+20%) signals stronger market conviction and amplifies the score.
# Low volume (-20%) signals weak participation and dampens the score.
def dsa_analyze_volume(data: pd.DataFrame) -> dict:
    avg_vol  = data["Volume"].mean()   # 30-day average volume
    last_vol = data["Volume"].iloc[-1] # most recent trading session's volume

    # Percentage change of the last volume relative to the 30-day average
    chg_pct  = (last_vol - avg_vol) / avg_vol * 100

    return {
        "avg_volume_30d":    int(avg_vol),
        "last_day_volume":   int(last_vol),
        "volume_change_pct": round(float(chg_pct), 2),
        # Ternary signal: high if >20% above average, low if >20% below, normal otherwise
        "signal": "high" if chg_pct > 20 else "low" if chg_pct < -20 else "normal",
    }


# ── FUNCTION: dsa_calculate_alpha ────────────────────────────────────────────
# Computes the stock's alpha: the difference between the stock's total return
# and the S&P 500 return over the same 30-day period.
# Positive alpha = stock outperformed the market; negative = underperformed.
def dsa_calculate_alpha(stock_data: pd.DataFrame, sp500_data: pd.DataFrame) -> dict:
    # Total return = (final price / initial price - 1) × 100
    stock_ret = (stock_data["Close"].iloc[-1] / stock_data["Close"].iloc[0] - 1) * 100
    sp500_ret = (sp500_data["Close"].iloc[-1] / sp500_data["Close"].iloc[0] - 1) * 100
    alpha     = stock_ret - sp500_ret  # excess return over the benchmark

    return {
        "stock_return_30d_pct": round(float(stock_ret), 2),
        "sp500_return_30d_pct": round(float(sp500_ret), 2),
        "alpha_pct":            round(float(alpha), 2),
        "beat_market":          bool(alpha > 0),  # True if the stock beat the S&P 500
    }


# ── FUNCTION: dsa_make_decision ───────────────────────────────────────────────
# Decision engine: combines the three indicators into a weighted composite score.
#
# Base formula:
#   score = (avg_sentiment × 0.45) + (sign(slope_pct) × 0.35) + (sign(alpha) × 0.20)
#
# Volume modifier:
#   volume "high"  → score × 1.15  (amplifies the signal — stronger conviction)
#   volume "low"   → score × 0.85  (dampens the signal — weaker conviction)
#
# Decision rule:
#   score > +0.25  → BUY
#   score < -0.25  → SELL
#   in between     → HOLD
#
# Why np.sign()? Only the direction of alpha and trend matters (positive/negative),
# not the magnitude — prevents highly volatile assets from dominating the calculation.
def dsa_make_decision(analyzed_articles: list, trend: dict, volume: dict, alpha: dict) -> dict:
    # Collects the numeric score of each article (+1, 0 or -1)
    scores    = [a["score"] for a in analyzed_articles]
    # Average sentiment: ranges from -1.0 (all negative) to +1.0 (all positive)
    avg_score = float(np.mean(scores)) if scores else 0.0

    slope_pct = trend["slope_pct_per_day"]
    alpha_val = alpha["alpha_pct"]
    vol_sig   = volume["signal"]

    # Weighted composite score: sentiment 45%, trend 35%, alpha 20%
    # np.sign() returns -1, 0 or +1 — normalizes the contribution of trend and alpha
    composite = (avg_score          * 0.45) + \
                (np.sign(slope_pct) * 0.35) + \
                (np.sign(alpha_val) * 0.20)

    # Applies the volume modifier: amplifies or dampens the final score
    if vol_sig == "high":
        composite *= 1.15
    elif vol_sig == "low":
        composite *= 0.85

    # Determines the decision and computes confidence based on distance from the threshold
    if composite > BUY_THRESHOLD:
        decision   = "BUY"
        confidence = min(abs(composite) * 100, 95)  # capped at 95% to avoid absolute certainty
    elif composite < SELL_THRESHOLD:
        decision   = "SELL"
        confidence = min(abs(composite) * 100, 95)
    else:
        decision   = "HOLD"
        # For HOLD: confidence is higher when the score is near zero (center of the neutral zone)
        confidence = max(0, 100 - abs(composite) * 200)

    log.info(f"Decision engine | composite={composite:.4f} | decision={decision} | confidence={confidence:.1f}%")
    return {
        "decision":        decision,
        "composite_score": round(composite, 4),
        "confidence_pct":  round(confidence, 1),
        "avg_sentiment":   round(avg_score, 4),
        "factors": {
            "sentiment":   avg_score,
            "price_trend": slope_pct,
            "alpha_sp500": alpha_val,
            "volume":      vol_sig,
        },
    }


# ── FUNCTION: dsa_build_news_table_rows ──────────────────────────────────────
# Builds the HTML table rows for the news section of the report.
# Applies HTML character escaping to prevent XSS.
# Uses alternating row colors (zebra striping) for better readability.
def dsa_build_news_table_rows(analyzed_articles: list) -> str:
    # Color and label map for each sentiment category
    badge_map = {
        "positive": ("#16a34a", "POSITIVE"),
        "negative": ("#dc2626", "NEGATIVE"),
        "neutral":  ("#64748b", "NEUTRAL"),
    }
    rows = ""
    for i, a in enumerate(analyzed_articles):
        cat   = a["category"]
        color, label = badge_map.get(cat, ("#333", cat.upper()))

        # HTML-escape to prevent tag injection from news content
        title         = a["title"].replace("<", "&lt;").replace(">", "&gt;")
        justification = a["justification"].replace("<", "&lt;").replace(">", "&gt;")
        source        = a["source"].replace("<", "&lt;").replace(">", "&gt;")
        pub           = a.get("published_at", "")[:10]  # date only (YYYY-MM-DD)
        url           = a.get("url", "#")

        # Alternating background between even and odd rows (zebra striping)
        bg            = "#f8fafc" if i % 2 == 0 else "#ffffff"
        rows += (
            f'<tr style="background:{bg}">'
            f'<td><a href="{url}" target="_blank" class="news-link">{title}</a></td>'
            f'<td><span class="badge" style="background:{color}22;color:{color}">{label}</span></td>'
            f'<td class="just-text">{justification}</td>'
            f'<td class="source-text">{source}</td>'
            f'<td class="date-text">{pub}</td>'
            f'</tr>\n'
        )
    return rows


# ── FUNCTION: dsa_generate_outputs ───────────────────────────────────────────
# Generates all output artifacts from the analysis:
#   - Interactive HTML report with charts (Chart.js)
#   - JSON file with all structured data
#   - CSV of analyzed news articles
#   - CSV of historical prices
# Returns a dict with the paths of the generated files.
def dsa_generate_outputs(ticker, company, analyzed_articles, stock_data, trend, volume, alpha, result, timestamp) -> dict:
    # Base filename: TICKER_YYYYMMDD_HHMMSS
    base  = ticker + "_" + timestamp
    paths = {
        "html":       "outputs/report_" + base + ".html",
        "json":       "outputs/data_"   + base + ".json",
        "news_csv":   "outputs/news_"   + base + ".csv",
        "prices_csv": "outputs/prices_" + base + ".csv",
    }

    # Builds the price list with cumulative daily return relative to the first day
    first_close = stock_data["stock"]["Close"].iloc[0]
    prices_list = []
    for idx, row in stock_data["stock"].iterrows():
        prices_list.append({
            "date":             str(idx.date()),
            "close":            round(float(row["Close"]), 4),
            "volume":           int(row["Volume"]),
            # Cumulative return: how many % gained/lost since the first day
            "daily_return_pct": round(float(row["Close"]) / float(first_close) * 100 - 100, 4),
        })

    # Custom serializer for NumPy types that the standard json module does not support
    def _json_safe(o):
        if isinstance(o, (np.bool_,)):       return bool(o)
        if isinstance(o, (np.integer,)):     return int(o)
        if isinstance(o, (np.floating,)):    return float(o)
        if isinstance(o, (np.ndarray,)):     return o.tolist()
        return str(o)

    # Saves the JSON with the full analysis data structure
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "ticker":    ticker,
                "company":   company,
                "timestamp": timestamp,
                "llm_used":  "ollama/" + LLM_MODEL,
            },
            "decision": result,
            "news":     list(analyzed_articles),
            "financial_indicators": {
                "trend":       trend,
                "volume":      volume,
                "alpha_sp500": alpha,
            },
            "prices_30d": prices_list,
        }, f, indent=2, ensure_ascii=False, default=_json_safe)

    # CSVs: analyzed news and historical prices for spreadsheet analysis
    pd.DataFrame(analyzed_articles).to_csv(paths["news_csv"],   index=False)
    pd.DataFrame(prices_list).to_csv(paths["prices_csv"], index=False)

    # ── HTML generation ───────────────────────────────────────────────────────
    # Decision badge colors: green=BUY, red=SELL, amber=HOLD
    decision_colors = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#d97706"}
    decision_color  = decision_colors.get(result["decision"], "#2563eb")

    # Sentiment counts for the summary cards and the donut chart
    pos   = sum(1 for a in analyzed_articles if a["category"] == "positive")
    neu   = sum(1 for a in analyzed_articles if a["category"] == "neutral")
    neg   = sum(1 for a in analyzed_articles if a["category"] == "negative")
    total = len(analyzed_articles)

    pos_pct = round(pos / total * 100, 1) if total else 0
    neu_pct = round(neu / total * 100, 1) if total else 0
    neg_pct = round(neg / total * 100, 1) if total else 0

    # Serialized data for injection into Chart.js charts via f-string
    prices_js  = [round(r["close"],  2) for r in prices_list]
    dates_js   = [r["date"]            for r in prices_list]
    volumes_js = [r["volume"]          for r in prices_list]

    # Cumulative return for the stock and S&P 500 for the comparison chart
    stock_cum_js = [r["daily_return_pct"] for r in prices_list]
    sp500_first  = stock_data["sp500"]["Close"].iloc[0]
    sp500_cum_js = [
        round(float(c) / float(sp500_first) * 100 - 100, 4)
        for c in stock_data["sp500"]["Close"]
    ]

    # Aligns series lengths — stock and S&P 500 may have different trading dates
    # due to market-specific holidays
    min_len      = min(len(dates_js), len(sp500_cum_js))
    dates_js     = dates_js[:min_len]
    prices_js    = prices_js[:min_len]
    volumes_js   = volumes_js[:min_len]
    stock_cum_js = stock_cum_js[:min_len]
    sp500_cum_js = sp500_cum_js[:min_len]

    # Trend line for the price chart: linear regression recomputed on the aligned prices
    # (may differ slightly from the main indicator calculation)
    xa = np.arange(len(prices_js))
    sl, ic, _, _, _ = stats.linregress(xa, prices_js)
    trend_line    = [round(ic + sl * float(xi), 2) for xi in xa]

    # Flat horizontal line at the 30-day average volume for the volume bar chart
    avg_vol_line  = [volume["avg_volume_30d"]] * len(dates_js)

    rows_html = dsa_build_news_table_rows(analyzed_articles)

    # Helper variables for the HTML template (conditional colors and dynamic text)
    last_price    = prices_js[-1] if prices_js else 0
    trend_color   = "#16a34a" if trend["slope_pct_per_day"] > 0 else "#dc2626"
    alpha_color   = "#16a34a" if alpha["beat_market"] else "#dc2626"
    sent_color    = "#16a34a" if result["avg_sentiment"] > 0.1 else "#dc2626" if result["avg_sentiment"] < -0.1 else "#64748b"
    ret_color     = "#16a34a" if alpha["stock_return_30d_pct"] >= 0 else "#dc2626"
    vol_mod       = "×1.15 (amplified)" if volume["signal"] == "high" else "×0.85 (dampened)" if volume["signal"] == "low" else "×1.00 (neutral)"
    vol_insight   = ("High volume confirms the price movement — the signal is amplified by 15%."
                     if volume["signal"] == "high" else
                     "Low volume suggests weak market conviction — the signal is dampened by 15%."
                     if volume["signal"] == "low" else
                     "Normal volume — no modification applied to the composite score.")
    beat_txt      = "outperforming" if alpha["beat_market"] else "underperforming"

    # Individual contributions of each factor to the composite score (for the rationale table)
    avg_s         = result["avg_sentiment"]
    slope_pct     = trend["slope_pct_per_day"]
    alpha_val     = alpha["alpha_pct"]
    sent_contrib  = round(avg_s * 0.45, 4)
    trend_contrib = round(float(np.sign(slope_pct)) * 0.35, 4)
    alpha_contrib = round(float(np.sign(alpha_val)) * 0.20, 4)
    raw_composite = round(sent_contrib + trend_contrib + alpha_contrib, 4)

    # Human-readable date string for display in the report
    dt_display    = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%B %d, %Y at %H:%M")
    conf_w        = min(result["confidence_pct"], 100)  # caps the confidence bar at 100%

    # Progress bar widths for the factor contribution bars in the hero card
    sc_abs_sent   = min(abs(sent_contrib)  * 100, 100)
    sc_abs_trend  = min(abs(trend_contrib) * 100, 100)
    sc_abs_alpha  = min(abs(alpha_contrib) * 100, 100)

    # Contribution bar colors: green if positive, red if negative
    sc_col_sent   = "#16a34a" if sent_contrib  >= 0 else "#dc2626"
    sc_col_trend  = "#16a34a" if trend_contrib >= 0 else "#dc2626"
    sc_col_alpha  = "#16a34a" if alpha_contrib >= 0 else "#dc2626"

    # Decision rule label used in the rationale table
    dec_label     = "> +0.25" if result["decision"] == "BUY" else "< -0.25" if result["decision"] == "SELL" else "between -0.25 and +0.25"

    # Full HTML template: inline CSS + Chart.js + data injected via f-string
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Investment Robot — {company} ({ticker})</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#f1f5f9; --card:#fff; --hdr:#0f172a;
    --text:#1e293b; --muted:#64748b; --border:#e2e8f0;
    --buy:#16a34a; --sell:#dc2626; --hold:#d97706;
    --blue:#2563eb; --indigo:#6366f1;
    --r:12px; --sh:0 1px 3px rgba(0,0,0,.07),0 4px 14px rgba(0,0,0,.06);
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.55}}

  /* Header */
  header{{background:var(--hdr);color:#f8fafc;padding:16px 32px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100;box-shadow:0 2px 10px rgba(0,0,0,.35)}}
  .hbrand{{font-size:1.05em;font-weight:700;letter-spacing:.2px}}.hbrand span{{color:#60a5fa}}
  .hmeta{{font-size:.78em;color:#94a3b8;text-align:right}}
  .pill{{display:inline-flex;align-items:center;gap:5px;background:#064e3b;color:#6ee7b7;border-radius:999px;padding:3px 10px;font-size:.75em;font-weight:700}}

  /* Layout */
  main{{max-width:1200px;margin:28px auto;padding:0 20px 60px}}
  section{{margin-bottom:22px}}

  /* Cards */
  .card{{background:var(--card);border-radius:var(--r);padding:26px 30px;box-shadow:var(--sh);border:1px solid var(--border)}}
  .ctitle{{font-size:1.08em;font-weight:700;margin-bottom:5px;display:flex;align-items:center;gap:9px}}
  .ctitle .ic{{font-size:1.15em}}
  .csub{{font-size:.84em;color:var(--muted);margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--border)}}

  /* Decision hero */
  .hero{{background:var(--hdr);border-radius:var(--r);padding:32px;color:#f8fafc;box-shadow:var(--sh);display:grid;grid-template-columns:auto 1fr;gap:30px;align-items:start}}
  .dbadge{{font-size:2.8em;font-weight:900;padding:18px 42px;border-radius:10px;color:#fff;background:{decision_color};letter-spacing:3px;text-align:center;box-shadow:0 4px 20px {decision_color}55}}
  .dmeta{{color:#94a3b8;font-size:.88em;margin-top:8px;text-align:center}}
  .htitle{{font-size:1.35em;font-weight:700;margin-bottom:4px}}
  .hsub{{color:#94a3b8;font-size:.87em;margin-bottom:14px}}

  /* Confidence bar */
  .clabel{{font-size:.82em;color:#94a3b8;margin-bottom:5px}}
  .cbarbg{{background:#1e293b;border-radius:999px;height:10px;overflow:hidden;margin-bottom:14px}}
  .cbarfill{{height:100%;border-radius:999px;background:{decision_color}}}

  /* Factor bars */
  .factors{{display:flex;flex-direction:column;gap:9px}}
  .frow{{display:flex;align-items:center;gap:10px;font-size:.83em}}
  .fname{{width:130px;color:#94a3b8;flex-shrink:0}}
  .fbarbg{{flex:1;background:#1e293b;border-radius:999px;height:7px;overflow:hidden}}
  .fbar{{height:100%;border-radius:999px}}
  .fval{{width:58px;text-align:right;color:#e2e8f0;font-weight:600}}

  /* Metrics row */
  .mgrid{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:22px}}
  .mc{{background:var(--card);border-radius:var(--r);padding:18px 22px;box-shadow:var(--sh);border:1px solid var(--border);border-left:4px solid {decision_color}}}
  .mlbl{{font-size:.76em;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:5px}}
  .mval{{font-size:1.75em;font-weight:800;line-height:1}}
  .mnote{{font-size:.76em;color:var(--muted);margin-top:4px}}

  /* Insight box */
  .insight{{background:#f0f9ff;border-left:4px solid #0ea5e9;border-radius:0 8px 8px 0;padding:13px 17px;margin-top:14px;font-size:.86em;color:#0c4a6e}}
  .insight strong{{display:block;margin-bottom:4px;color:#0369a1}}

  /* Chart wrappers */
  .cw{{position:relative;height:280px;margin:18px 0 4px}}
  .cw-sm{{position:relative;height:230px;margin:18px 0 4px}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:22px;align-items:start}}

  /* Sentiment cards */
  .sgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin-bottom:18px}}
  .sc{{border-radius:10px;padding:16px;text-align:center}}
  .sv{{font-size:2.1em;font-weight:800;line-height:1}}
  .sl{{font-size:.78em;margin-top:4px;font-weight:600}}

  /* Table */
  .twrap{{overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.875em}}
  thead th{{background:#1e293b;color:#f8fafc;padding:11px 13px;text-align:left;font-weight:600;letter-spacing:.3px}}
  tbody td{{padding:9px 13px;border-bottom:1px solid var(--border);vertical-align:top}}
  tbody tr:last-child td{{border-bottom:none}}
  .news-link{{color:var(--text);text-decoration:none}}
  .news-link:hover{{color:var(--blue);text-decoration:underline}}
  .badge{{display:inline-block;padding:3px 9px;border-radius:999px;font-size:.78em;font-weight:700;letter-spacing:.3px}}
  .just-text{{color:#475569;max-width:270px}}
  .source-text{{color:var(--muted);font-size:.83em;white-space:nowrap}}
  .date-text{{color:#94a3b8;font-size:.78em;white-space:nowrap}}

  /* Rationale table */
  .rat td{{padding:10px 13px;border-bottom:1px solid var(--border);font-size:.88em}}
  .rat tr:last-child td{{border-bottom:none}}

  /* Disclaimer / Footer */
  .disc{{background:#fffbeb;border:1px solid #fbbf24;border-radius:10px;padding:15px 20px;font-size:.84em;color:#92400e;margin-top:22px}}
  footer{{text-align:center;color:#94a3b8;font-size:.76em;padding:22px;border-top:1px solid var(--border);margin-top:36px}}

  @media(max-width:768px){{
    .mgrid{{grid-template-columns:repeat(2,1fr)}}
    .hero{{grid-template-columns:1fr}}
    .two-col{{grid-template-columns:1fr}}
    header{{flex-direction:column;gap:8px;text-align:center}}
  }}
</style>
</head>
<body>

<header>
  <div>
    <div class="hbrand">AI Investment Robot <span>| {company} ({ticker})</span></div>
    <div class="hmeta">{dt_display}</div>
  </div>
  <div style="text-align:right">
    <div class="pill">&#128274; 100% Local — Llama 3.2 via Ollama</div>
    <div class="hmeta" style="margin-top:4px">Data: NewsAPI + Yahoo Finance</div>
  </div>
</header>

<main>

<!-- ── DECISION HERO ───────────────────────────────────────────────────── -->
<section>
  <div class="hero">
    <div>
      <div class="dbadge">{result['decision']}</div>
      <div class="dmeta">Confidence: {result['confidence_pct']}% &nbsp;|&nbsp; Score: {result['composite_score']:+.4f}</div>
    </div>
    <div>
      <div class="htitle">{company} ({ticker}) — Investment Signal</div>
      <div class="hsub">Generated on {dt_display} · Llama 3.2 running locally via Ollama · No data sent to external AI APIs</div>
      <div class="clabel">Decision confidence: {result['confidence_pct']}%</div>
      <div class="cbarbg"><div class="cbarfill" style="width:{conf_w:.1f}%"></div></div>
      <div class="factors">
        <div class="clabel" style="margin-bottom:2px">Score breakdown (weights: Sentiment 45% · Trend 35% · Alpha 20%):</div>
        <div class="frow">
          <span class="fname">Sentiment (45%)</span>
          <div class="fbarbg"><div class="fbar" style="width:{sc_abs_sent:.0f}%;background:{sc_col_sent}"></div></div>
          <span class="fval">{sent_contrib:+.3f}</span>
        </div>
        <div class="frow">
          <span class="fname">Price Trend (35%)</span>
          <div class="fbarbg"><div class="fbar" style="width:{sc_abs_trend:.0f}%;background:{sc_col_trend}"></div></div>
          <span class="fval">{trend_contrib:+.3f}</span>
        </div>
        <div class="frow">
          <span class="fname">Alpha S&amp;P500 (20%)</span>
          <div class="fbarbg"><div class="fbar" style="width:{sc_abs_alpha:.0f}%;background:{sc_col_alpha}"></div></div>
          <span class="fval">{alpha_contrib:+.3f}</span>
        </div>
        <div class="frow">
          <span class="fname">Volume Modifier</span>
          <span style="color:#94a3b8;font-size:.82em">{volume['signal'].upper()} ({volume['volume_change_pct']:+.1f}% vs avg) &#8594; {vol_mod}</span>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ── KEY METRICS ────────────────────────────────────────────────────── -->
<div class="mgrid">
  <div class="mc">
    <div class="mlbl">Last Close Price</div>
    <div class="mval">${last_price:.2f}</div>
    <div class="mnote">{ticker} · 30-day period</div>
  </div>
  <div class="mc">
    <div class="mlbl">30-Day Return</div>
    <div class="mval" style="color:{ret_color}">{alpha['stock_return_30d_pct']:+.2f}%</div>
    <div class="mnote">S&amp;P500: {alpha['sp500_return_30d_pct']:+.2f}%</div>
  </div>
  <div class="mc">
    <div class="mlbl">Alpha vs S&amp;P 500</div>
    <div class="mval" style="color:{alpha_color}">{alpha['alpha_pct']:+.2f}%</div>
    <div class="mnote">{'Outperforming' if alpha['beat_market'] else 'Underperforming'} the market</div>
  </div>
  <div class="mc">
    <div class="mlbl">Avg Sentiment Score</div>
    <div class="mval" style="color:{sent_color}">{avg_s:+.3f}</div>
    <div class="mnote">{total} articles · scale −1.0 to +1.0</div>
  </div>
</div>

<!-- ── SENTIMENT ANALYSIS ─────────────────────────────────────────────── -->
<section class="card">
  <div class="ctitle"><span class="ic">&#128240;</span> News Sentiment Analysis</div>
  <div class="csub">
    Llama 3.2 (running locally) classified {total} recent news articles about <strong>{company}</strong> into positive, negative, or neutral, with a short justification per article. This signal carries <strong>45% weight</strong> in the final decision — the largest single factor.
  </div>
  <div class="sgrid">
    <div class="sc" style="background:#f0fdf4">
      <div class="sv" style="color:#16a34a">{pos}</div>
      <div class="sl" style="color:#15803d">POSITIVE &nbsp;·&nbsp; {pos_pct}%</div>
    </div>
    <div class="sc" style="background:#f8fafc">
      <div class="sv" style="color:#64748b">{neu}</div>
      <div class="sl" style="color:#475569">NEUTRAL &nbsp;·&nbsp; {neu_pct}%</div>
    </div>
    <div class="sc" style="background:#fef2f2">
      <div class="sv" style="color:#dc2626">{neg}</div>
      <div class="sl" style="color:#b91c1c">NEGATIVE &nbsp;·&nbsp; {neg_pct}%</div>
    </div>
  </div>
  <div class="two-col">
    <div><div class="cw-sm"><canvas id="pieChart"></canvas></div></div>
    <div>
      <div class="insight">
        <strong>How to read this chart</strong>
        The donut shows the proportion of positive, neutral and negative articles. A higher share of positive news pushes the average sentiment score toward <strong>+1.0</strong>, contributing positively to the buy signal.
        <br><br>
        <strong>Average sentiment score: {avg_s:+.3f}</strong> (scale −1.0 to +1.0). Values above +0.1 are considered bullish; below −0.1 are bearish. Each article receives +1 (positive), 0 (neutral), or −1 (negative); the score is their mean. This contributes <strong>{sent_contrib:+.3f}</strong> to the composite (after applying the 45% weight).
      </div>
    </div>
  </div>
</section>

<!-- ── PRICE HISTORY & TREND ──────────────────────────────────────────── -->
<section class="card">
  <div class="ctitle"><span class="ic">&#128200;</span> Price History &amp; Trend — {ticker} (30 Days)</div>
  <div class="csub">
    Daily closing price over the last 30 trading days with a <strong>linear regression trend line</strong>. The trend direction and strength carry <strong>35% weight</strong> in the final decision.
  </div>
  <div class="cw"><canvas id="priceChart"></canvas></div>
  <div class="insight">
    <strong>How to read this chart</strong>
    The <span style="color:#2563eb;font-weight:700">blue line</span> shows actual daily closing prices. The <span style="color:#dc2626;font-weight:700">dashed red line</span> is the <em>linear regression</em> — a statistical best-fit line that reveals the underlying direction of price, filtering out day-to-day noise.
    <br><br>
    <strong>Trend: {trend['trend'].upper()} ({trend['slope_pct_per_day']:+.4f}%/day)</strong> — The slope represents how much the price shifts daily as a percentage of its 30-day mean. A positive slope contributes <strong>+0.35</strong> to the score; negative contributes <strong>−0.35</strong>.
    <br><br>
    <strong>R² = {trend['r_squared']:.4f}</strong> — The coefficient of determination measures how tightly prices follow the trend line. R² near 1.0 = consistent, reliable trend; R² near 0 = noisy, unpredictable — use this value to gauge how much to trust the trend signal.
  </div>
</section>

<!-- ── CUMULATIVE RETURN VS S&P 500 ───────────────────────────────────── -->
<section class="card">
  <div class="ctitle"><span class="ic">&#9889;</span> Cumulative Return vs S&amp;P 500 Benchmark (30 Days)</div>
  <div class="csub">
    Head-to-head performance: how much {ticker} returned compared to the S&amp;P 500 index over the same period. The gap between the lines is the <strong>Alpha</strong>, which carries <strong>20% weight</strong>.
  </div>
  <div class="cw"><canvas id="returnChart"></canvas></div>
  <div class="insight">
    <strong>How to read this chart</strong>
    Both lines start at <strong>0%</strong> on the first trading day and show cumulative percentage return. When the <span style="color:#2563eb;font-weight:700">{ticker} line</span> is above the <span style="color:#94a3b8;font-weight:700">S&amp;P 500 line</span>, the stock is <em>outperforming</em> the market.
    <br><br>
    <strong>Alpha: {alpha['alpha_pct']:+.2f}%</strong> — {ticker} returned <strong>{alpha['stock_return_30d_pct']:+.2f}%</strong> vs the S&amp;P 500's <strong>{alpha['sp500_return_30d_pct']:+.2f}%</strong>, meaning the stock is <em>{beat_txt}</em> the market by {abs(alpha['alpha_pct']):.2f}%. Positive alpha contributes <strong>+0.20</strong> to the score; negative contributes <strong>−0.20</strong>.
  </div>
</section>

<!-- ── VOLUME ANALYSIS ────────────────────────────────────────────────── -->
<section class="card">
  <div class="ctitle"><span class="ic">&#128202;</span> Trading Volume (30 Days)</div>
  <div class="csub">
    Daily number of shares traded, with the 30-day average shown as a reference line. Volume acts as a <strong>multiplier (±15%)</strong> on the composite score — it does not drive the decision alone but amplifies or dampens the other signals.
  </div>
  <div class="cw"><canvas id="volChart"></canvas></div>
  <div class="insight">
    <strong>How to read this chart</strong>
    The <span style="color:#6366f1;font-weight:700">bars</span> show daily trading volume. The <span style="color:#f97316;font-weight:700">dashed orange line</span> is the 30-day average ({volume['avg_volume_30d']:,} shares/day). Volume measures market conviction: a price move on high volume is more credible than the same move on thin volume.
    <br><br>
    <strong>Volume Signal: {volume['signal'].upper()} ({volume['volume_change_pct']:+.1f}% vs 30-day average)</strong> — Last session: {volume['last_day_volume']:,} shares. {vol_insight} Applied modifier to composite score: <strong>{vol_mod}</strong>.
  </div>
</section>

<!-- ── DECISION RATIONALE ─────────────────────────────────────────────── -->
<section class="card">
  <div class="ctitle"><span class="ic">&#129518;</span> Decision Rationale — Full Score Calculation</div>
  <div class="csub">
    Step-by-step breakdown of how the composite score was computed. Decision rule: score &gt; +0.25 &#8594; BUY &nbsp;|&nbsp; score &lt; −0.25 &#8594; SELL &nbsp;|&nbsp; otherwise &#8594; HOLD.
  </div>
  <table class="rat">
    <tbody>
      <tr>
        <td style="font-weight:600;width:210px">Sentiment signal</td>
        <td>avg_score &times; 0.45 = {avg_s:+.4f} &times; 0.45 = <strong>{sent_contrib:+.4f}</strong></td>
        <td style="color:var(--muted);font-size:.84em">{pos} positive + {neu} neutral + {neg} negative = avg {avg_s:+.4f}</td>
      </tr>
      <tr>
        <td style="font-weight:600">Price Trend signal</td>
        <td>sign({slope_pct:+.4f}) &times; 0.35 = <strong>{trend_contrib:+.4f}</strong></td>
        <td style="color:var(--muted);font-size:.84em">Slope = {trend['slope']:+.6f} USD/day ({slope_pct:+.4f}%/day) · R² = {trend['r_squared']:.4f} · {trend['trend'].upper()}</td>
      </tr>
      <tr>
        <td style="font-weight:600">Alpha signal</td>
        <td>sign({alpha_val:+.4f}) &times; 0.20 = <strong>{alpha_contrib:+.4f}</strong></td>
        <td style="color:var(--muted);font-size:.84em">{ticker} {alpha['stock_return_30d_pct']:+.2f}% − S&amp;P500 {alpha['sp500_return_30d_pct']:+.2f}% = alpha {alpha_val:+.2f}%</td>
      </tr>
      <tr>
        <td style="font-weight:600">Raw composite</td>
        <td colspan="2">{sent_contrib:+.4f} + {trend_contrib:+.4f} + {alpha_contrib:+.4f} = <strong>{raw_composite:+.4f}</strong></td>
      </tr>
      <tr>
        <td style="font-weight:600">Volume modifier</td>
        <td colspan="2">Signal: <strong>{volume['signal'].upper()}</strong> &#8594; {vol_mod} &#8594; Final composite: <strong>{result['composite_score']:+.4f}</strong></td>
      </tr>
      <tr style="background:#f0fdf4">
        <td style="font-weight:700;color:{decision_color}">FINAL DECISION</td>
        <td style="font-weight:700;color:{decision_color}">{result['decision']} &nbsp; (score {result['composite_score']:+.4f} {dec_label})</td>
        <td style="color:var(--muted);font-size:.84em">Confidence: {result['confidence_pct']:.1f}%</td>
      </tr>
    </tbody>
  </table>
</section>

<!-- ── NEWS ARTICLES ──────────────────────────────────────────────────── -->
<section class="card">
  <div class="ctitle"><span class="ic">&#128478;</span> News Articles Analyzed ({total} articles)</div>
  <div class="csub">
    Each article (title + summary) was independently sent to Llama 3.2 with a structured prompt asking for a sentiment category and a short justification. Model temperature was set to 0.1 to maximise response consistency.
  </div>
  <div class="twrap">
    <table>
      <thead><tr><th>Title</th><th>Sentiment</th><th>LLM Justification</th><th>Source</th><th>Date</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</section>

<!-- ── DISCLAIMER ─────────────────────────────────────────────────────── -->
<div class="disc">
  &#9888;&#65039; <strong>Important Disclaimer:</strong> This report is produced by an automated quantitative analysis system for informational purposes only. It does <strong>not</strong> constitute financial advice, investment recommendation, or a solicitation to buy or sell any security. Past performance is not indicative of future results. Always consult a qualified financial advisor before making investment decisions.
</div>

</main>

<footer>AI Investment Robot &nbsp;|&nbsp; Developed by Andre Kim Scarton &nbsp;|&nbsp; {ticker} &nbsp;·&nbsp; {dt_display} &nbsp;·&nbsp; Llama 3.2 via Ollama (local)</footer>

<script>
// ── Data injected by Python via f-string ─────────────────────────────────────
const labels    = {json.dumps(dates_js)};    // trading session dates (X-axis)
const prices    = {json.dumps(prices_js)};   // closing prices
const trendLine = {json.dumps(trend_line)};  // linear regression line
const volumes   = {json.dumps(volumes_js)};  // daily volumes
const stockCum  = {json.dumps(stock_cum_js)};  // stock cumulative return
const sp500Cum  = {json.dumps(sp500_cum_js)};  // S&P 500 cumulative return
const avgVol    = {json.dumps(avg_vol_line)};  // flat 30-day average volume line

// Base configuration shared across all Chart.js charts
const base = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{ legend:{{ labels:{{ font:{{ size:11 }} }} }} }},
  scales:{{
    x:{{ grid:{{ color:'#f1f5f9' }}, ticks:{{ maxTicksLimit:8, font:{{ size:10 }} }} }},
    y:{{ grid:{{ color:'#f1f5f9' }}, ticks:{{ font:{{ size:10 }} }} }}
  }}
}};

// Chart 1: Closing price + trend line (linear regression)
new Chart(document.getElementById('priceChart'),{{
  type:'line',
  data:{{ labels, datasets:[
    {{ label:'{ticker} Close', data:prices, borderColor:'#2563eb', backgroundColor:'rgba(37,99,235,.07)',
       fill:true, tension:0.3, pointRadius:2, borderWidth:2 }},
    {{ label:'Trend (Linear Regression)', data:trendLine, borderColor:'#dc2626',
       borderDash:[6,3], pointRadius:0, fill:false, borderWidth:1.5 }}
  ]}},
  options:{{ ...base,
    plugins:{{ ...base.plugins, tooltip:{{ callbacks:{{ label: c => c.dataset.label+': $'+c.parsed.y.toFixed(2) }} }} }},
    scales:{{ ...base.scales, y:{{ ...base.scales.y, ticks:{{ ...base.scales.y.ticks, callback: v=>'$'+v }} }} }}
  }}
}});

// Chart 2: Cumulative return of the stock vs S&P 500 (both start at 0%)
new Chart(document.getElementById('returnChart'),{{
  type:'line',
  data:{{ labels, datasets:[
    {{ label:'{ticker} Cumulative Return', data:stockCum, borderColor:'#2563eb', backgroundColor:'rgba(37,99,235,.07)',
       fill:true, tension:0.3, pointRadius:2, borderWidth:2 }},
    {{ label:'S&P 500 Cumulative Return', data:sp500Cum, borderColor:'#94a3b8',
       borderDash:[4,2], pointRadius:0, fill:false, borderWidth:1.5 }}
  ]}},
  options:{{ ...base,
    plugins:{{ ...base.plugins, tooltip:{{ callbacks:{{ label: c => c.dataset.label+': '+c.parsed.y.toFixed(2)+'%' }} }} }},
    scales:{{ ...base.scales, y:{{ ...base.scales.y, ticks:{{ ...base.scales.y.ticks, callback: v=>v.toFixed(1)+'%' }} }} }}
  }}
}});

// Chart 3: Daily volume (bars) + 30-day average (dashed line)
new Chart(document.getElementById('volChart'),{{
  type:'bar',
  data:{{ labels, datasets:[
    {{ label:'Daily Volume', data:volumes, backgroundColor:'rgba(99,102,241,.55)', borderColor:'rgba(99,102,241,.8)', borderWidth:1 }},
    {{ label:'30-Day Average', data:avgVol, type:'line', borderColor:'#f97316',
       borderDash:[5,3], pointRadius:0, fill:false, borderWidth:2 }}
  ]}},
  options:{{ ...base,
    plugins:{{ ...base.plugins, tooltip:{{ callbacks:{{ label: c => c.dataset.label+': '+Number(c.parsed.y).toLocaleString() }} }} }},
    scales:{{ ...base.scales, y:{{ ...base.scales.y, ticks:{{ ...base.scales.y.ticks, callback: v=>(v/1e6).toFixed(1)+'M' }} }} }}
  }}
}});

// Chart 4: Sentiment distribution donut (positive / neutral / negative)
new Chart(document.getElementById('pieChart'),{{
  type:'doughnut',
  data:{{ labels:['Positive','Neutral','Negative'],
    datasets:[{{ data:[{pos},{neu},{neg}], backgroundColor:['#16a34a','#94a3b8','#dc2626'],
      borderWidth:2, borderColor:'#fff', hoverOffset:8 }}]
  }},
  options:{{ responsive:true, maintainAspectRatio:false, cutout:'60%',
    plugins:{{ legend:{{ position:'bottom', labels:{{ padding:14, font:{{ size:11 }} }} }} }}
  }}
}});
</script>
</body>
</html>"""

    with open(paths["html"], "w", encoding="utf-8") as f:
        f.write(html)

    return paths


# ── FUNCTION: dsa_display_summary ────────────────────────────────────────────
# Prints a formatted summary of the analysis to the terminal at the end of the run,
# including sentiment counts, financial indicators and the final decision.
def dsa_display_summary(ticker, company, analyzed_articles, trend, volume, alpha, result, paths, timestamp):
    total = len(analyzed_articles)
    pos   = sum(1 for a in analyzed_articles if a["category"] == "positive")
    neu   = sum(1 for a in analyzed_articles if a["category"] == "neutral")
    neg   = sum(1 for a in analyzed_articles if a["category"] == "negative")

    # Local helper to compute percentage with protection against division by zero
    def pct(n):
        return str(round(n / total * 100, 1)) + "%" if total else "0%"

    dt   = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%m/%d/%Y %H:%M:%S")
    beat = "outperformed" if alpha["beat_market"] else "underperformed"

    print("")
    print("+==================================================================+")
    print("|              AI INVESTMENT ROBOT — ANALYSIS COMPLETE            |")
    print("|         LLM: Llama 3.2 via Ollama (local, free, private)        |")
    print("+==================================================================+")
    print(f"  Company : {company}   Ticker : {ticker}")
    print(f"  Date    : {dt}")
    print("+------------------------------------------------------------------+")
    print(f"  NEWS ANALYZED: {total}")
    print(f"  Positive : {pos}  ({pct(pos)})")
    print(f"  Neutral  : {neu}  ({pct(neu)})")
    print(f"  Negative : {neg}  ({pct(neg)})")
    print(f"  Average sentiment score: {result['avg_sentiment']:+.4f}")
    print("+------------------------------------------------------------------+")
    print(f"  Trend      : {trend['trend'].upper()}  ({trend['slope_pct_per_day']:+.3f}%/day | R2={trend['r_squared']:.2f})")
    print(f"  vs S&P500  : {alpha['alpha_pct']:+.2f}%  ({beat} the market)")
    print(f"  Volume     : {volume['signal'].upper()}  ({volume['volume_change_pct']:+.1f}% vs 30-day avg)")
    print("+------------------------------------------------------------------+")
    print(f"  *** DECISION: {result['decision']} ***   Confidence: {result['confidence_pct']:.1f}%   Score: {result['composite_score']:+.4f}")
    print("+------------------------------------------------------------------+")
    print("  DISCLAIMER: Educational project -- not financial advice.")
    print("+------------------------------------------------------------------+")
    print(f"  HTML  : {paths['html']}")
    print(f"  JSON  : {paths['json']}")
    print("  Log   : logs/investment_robot.log")
    print("+==================================================================+")
    print("")


# ── FUNCTION: main ────────────────────────────────────────────────────────────
# Main pipeline that orchestrates all steps in the correct sequence:
#   1. Ollama health check
#   2. Interactive menu (ticker + company name)
#   3. News collection (NewsAPI)
#   4. Async sentiment analysis (Llama 3.2)
#   5. Historical price data (yfinance)
#   6. Indicator calculation (trend, volume, alpha)
#   7. Decision engine (composite score)
#   8. Report generation (HTML, JSON, CSV)
#   9. Terminal summary display
async def main():
    # Step 1: Ensure Ollama is running and the model is available
    dsa_check_ollama()

    # Step 2: Collect ticker and company name via the validated interactive menu
    ticker, company = dsa_interactive_menu()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"Analysis started | ticker={ticker} | company={company} | llm=ollama/{LLM_MODEL}")

    # Step 3: Fetch recent news articles about the company via NewsAPI
    articles = dsa_collect_news(company)
    if not articles:
        # No news found: default decision is HOLD — no sentiment signal available
        print("\n[!] No news articles found. Setting decision to HOLD by default.")
        log.warning("No articles -- default decision: HOLD")
        articles = []

    # Step 4: Async sentiment analysis — all articles processed in parallel
    if articles:
        print(f"\n[~] Analyzing sentiment for {len(articles)} articles via Llama 3.2 (local async)...")
        print("    Local models are slower -- please wait (~30-90s)...\n")
        analyzed = await dsa_analyze_sentiment_async(articles, company)
    else:
        analyzed = []

    # Step 5: Download 30 days of price history for the stock and the S&P 500
    print(f"\n[~] Fetching 30-day stock data for {ticker}...")
    stock_data = dsa_collect_stock_data(ticker)

    # Step 6: Compute the three financial indicators
    trend  = dsa_calculate_trend(stock_data["stock"]["Close"])  # linear regression on price
    volume = dsa_analyze_volume(stock_data["stock"])             # comparison against average volume
    alpha  = dsa_calculate_alpha(stock_data["stock"], stock_data["sp500"])  # excess return vs S&P 500
    log.info(f"Indicators | slope={trend['slope_pct_per_day']}%/day | alpha={alpha['alpha_pct']}% | volume={volume['signal']}")

    # Step 7: Decision engine — combines sentiment + trend + alpha with weights
    result = dsa_make_decision(analyzed, trend, volume, alpha)
    log.info(f"Final decision | {result['decision']} | confidence={result['confidence_pct']}%")

    # Step 8: Generate all output artifacts (HTML, JSON, CSVs)
    print("\n[~] Generating outputs...")
    paths = dsa_generate_outputs(
        ticker, company, analyzed, stock_data,
        trend, volume, alpha, result, timestamp
    )
    log.info(f"Outputs generated | html={paths['html']} | json={paths['json']}")

    # Step 9: Print the formatted summary to the terminal with all results
    dsa_display_summary(
        ticker, company, analyzed,
        trend, volume, alpha, result, paths, timestamp
    )


# Entry point: asyncio.run() initializes the event loop and runs main()
# The event loop is required for the async Llama calls made via httpx
if __name__ == "__main__":
    asyncio.run(main())
