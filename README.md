# 🤖 AI Investment Robot

> A stock analysis system that combines **local Artificial Intelligence**,
> **technical price analysis** and **market benchmarking** to deliver
> a clear, data-driven recommendation: **BUY · HOLD · SELL**.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Llama 3.2](https://img.shields.io/badge/Llama_3.2-Local_AI-8B5CF6?style=flat)](https://ollama.com/)
[![yfinance](https://img.shields.io/badge/yfinance-Stock_Data-10B981?style=flat)](https://pypi.org/project/yfinance/)
[![NewsAPI](https://img.shields.io/badge/NewsAPI-News-F59E0B?style=flat)](https://newsapi.org/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-AAPL_Report-success?style=flat)](https://andrescarton.github.io/Ai-Investment/)

---

## 🌐 Live report demo

**See exactly what the robot generates** — this is a real report produced by running
the system locally on Apple (AAPL), rendered as a static page:

> 👉 **[andrescarton.github.io/Ai-Investment](https://andrescarton.github.io/Ai-Investment/)**

The report includes interactive charts, a full news sentiment breakdown,
price trend regression, cumulative return vs S&P 500, volume analysis,
and a step-by-step score calculation — all generated automatically after
a single terminal command.

**What this specific run found (AAPL · June 7, 2026):**
- Decision: **BUY** · Confidence: 56.8% · Score: +0.5678
- 30-day return: **+7.00%** vs S&P 500 **+0.25%** → Alpha: **+6.74%**
- Trend: **UPWARD** (+0.39%/day, R² = 0.85)
- Volume: **HIGH** (+35.5% vs average) → signal amplified ×1.15
- Sentiment: 1 positive · 5 neutral · 2 negative → score −0.125

---

## 💡 About this project

This project started as a personal challenge: learn AI by actually building
something with it — not just following tutorials, but solving a real problem
I care about.

I've been interested in financial markets for a while and kept asking myself:
*can I use a local LLM to read the news, cross it with price data, and get a
smarter signal before making an investment decision?*

This is my answer. A hands-on project where two things I genuinely enjoy —
**Artificial Intelligence** and **investing** — come together in a working system.

The robot runs **100% locally** (no external AI APIs), uses **Llama 3.2 via Ollama**
to keep your data private, and generates a polished HTML report with interactive
charts at the end of every analysis.

---

## 🧠 How it works

Three independent signals are weighted and combined into a single composite score:

| Signal                  | Weight | What it captures                                       |
|-------------------------|--------|--------------------------------------------------------|
| 💬 News sentiment       | 45%    | Qualitative info before it's priced into the market    |
| 📈 Price trend          | 35%    | 30-day momentum via linear regression                  |
| ⚡ Alpha vs S&P 500     | 20%    | Relative performance against the broad market          |

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
  → News collection     (NewsAPI, up to 20 articles)
  → Sentiment analysis  (Llama 3.2 via Ollama, async parallel)
  → Financial data      (yfinance, 30-day history + S&P 500 benchmark)
  → Indicators          (trend, alpha, volume)
  → Decision engine     (weighted composite score)
  → Report generation   (HTML + JSON + CSV + log)
```

Sentiment analysis uses `asyncio.gather()` with a semaphore (3 concurrent calls)
to process all articles in parallel — reducing analysis time from ~15 min to ~60 sec.

---

## 📊 Generated outputs

Each run automatically creates:

| File                        | Format | Contents                                                                       |
|-----------------------------|--------|--------------------------------------------------------------------------------|
| `report_{TICKER}_{TS}.html` | HTML   | 4 interactive Chart.js graphs, metric cards, news table, full score rationale  |
| `data_{TICKER}_{TS}.json`   | JSON   | Decision, indicators, news scores, 30-day price history                        |
| `news_{TICKER}_{TS}.csv`    | CSV    | News articles with sentiment, score and LLM justification per article          |
| `prices_{TICKER}_{TS}.csv`  | CSV    | Daily close, volume and cumulative return                                      |
| `investment_robot.log`      | LOG    | Persistent append-only log across all runs                                     |

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

# 4. Pull the model and start Ollama
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

> Brazilian stocks (B3) are not currently supported due to limited
> Portuguese-language coverage in NewsAPI's free tier.
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
