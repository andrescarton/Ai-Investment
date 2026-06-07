# 🤖 AI Investment Robot

> A stock analysis system that combines **local Artificial Intelligence**,
> **technical price analysis** and **market benchmarking** to deliver
> a clear, data-driven recommendation: **BUY · HOLD · SELL**.

---

## 💡 About this project

This project started as a personal challenge: learn AI by actually building
something with it — not just following tutorials, but solving a real problem
I care about.

I've always been interested in financial markets, and I kept asking myself:
*can I use a local LLM to read news, cross it with price data, and get a
smarter signal before making an investment decision?*

This is my answer to that question. A hands-on project where two things
I genuinely enjoy — **Artificial Intelligence** and **investing** — come
together in a working system.

The robot runs **100% locally** (no external AI APIs), uses **Llama 3.2
via Ollama** for privacy, and generates a full HTML report with interactive
charts at the end of every analysis.

---

## 🧠 How it works

Three independent signals are weighted and combined into a single score:

| Signal                  | Weight | What it captures                                      |
|-------------------------|--------|-------------------------------------------------------|
| 💬 News sentiment       | 45%    | Qualitative info before it's priced into the market   |
| 📈 Price trend          | 35%    | 30-day momentum via linear regression                 |
| ⚡ Alpha vs S&P 500 | 20%    | Relative performance against the broad market         |

**Volume** acts as a modifier (±15%) — high volume amplifies the signal,
low volume attenuates it.

**Final decision thresholds:**
- Score > 0.25   → **BUY**
- Score between ±0.25 → **HOLD**
- Score < -0.25  → **SELL**

---

## ⚙️ Execution pipeline

```
Interactive menu
  → News collection (NewsAPI, up to 20 articles)
  → Sentiment analysis (Llama 3.2 via Ollama, async)
  → Financial data (yfinance, 30-day history + S&P 500 benchmark)
  → Indicators (trend, alpha, volume)
  → Decision engine (weighted score)
  → Report generation (HTML + JSON + CSV + log)
```

Sentiment analysis uses `asyncio.gather()` with a semaphore to process
all articles in parallel — reducing analysis time from ~15 min to ~60 sec.

---

## 📊 Generated outputs

Each run automatically creates:

| File                        | Format | Contents                                          |
|-----------------------------|--------|---------------------------------------------------|
| `report_{TICKER}_{TS}.html` | HTML   | 4 interactive Chart.js graphs, metric cards, news table, score rationale |
| `data_{TICKER}_{TS}.json`   | JSON   | Full analysis data: decision, indicators, news scores, price history |
| `news_{TICKER}_{TS}.csv`    | CSV    | News articles with sentiment, score and LLM justification |
| `prices_{TICKER}_{TS}.csv`  | CSV    | Daily close, volume and cumulative return         |
| `investment_robot.log`      | LOG    | Persistent append-only log across all runs        |

---

## 🚀 Quick start

```bash
# 1. Create and activate the environment
conda create --name ai-robot python=3.12
conda activate ai-robot

# 2. Install dependencies
pip install numpy pandas scipy yfinance newsapi-python httpx python-dotenv

# 3. Add your NewsAPI key
echo NEWSAPI_KEY=your-key-here > .env

# 4. Start Ollama (make sure Llama 3.2 is pulled)
ollama pull llama3.2
ollama serve

# 5. Run
python projeto10-llama-fixed.py
```

The system will prompt for a **ticker** (e.g. `AAPL`, `NVDA`, `TSLA`) and
a **company name** for the news search. Everything else runs automatically.

**Requirements:** Python 3.12 · Ollama + Llama 3.2 (~4 GB) · Free NewsAPI key

---

## 📈 Supported stocks

US-listed companies on **NYSE** and **NASDAQ**:
`AAPL` `MSFT` `GOOGL` `TSLA` `AMZN` `NVDA` `META` `NFLX` and similar.

> Brazilian stocks (B3) are not supported in the current version due to
> limited Portuguese-language coverage in NewsAPI's free tier.
> RSS integration with Infomoney / Valor Econômico is on the roadmap.

---

## 🗺️ Roadmap

- [ ] Brazilian market (B3) via RSS news feeds
- [ ] Multi-ticker analysis with portfolio ranking
- [ ] Historical backtesting vs buy-and-hold
- [ ] Automated daily scheduling with email / Telegram delivery
- [ ] Real-time dashboard (Streamlit or FastAPI + React)
- [ ] Multi-LLM voting for more robust sentiment classification
- [ ] Brokerage integration (Alpaca API) for automated order execution

---

## 🛠️ Tech stack

`Python 3.12` · `Llama 3.2` · `Ollama` · `yfinance` · `NewsAPI`
`asyncio` · `httpx` · `scipy` · `pandas` · `numpy` · `Chart.js`

---

## ⚠️ Disclaimer

This project is built for **educational and personal learning purposes**.
It does not constitute financial advice, investment recommendation, or
solicitation to buy or sell any security. Past performance is not indicative
of future results. Always consult a qualified financial advisor before
making investment decisions.
