# CLAUDE.md — Robô Investidor com IA (Projeto 10 DSA)

**Curso:** Engenharia Financeira com Inteligência Artificial — Data Science Academy  
**Capítulo:** 17 — LLMs para Análise Financeira  
**Autor:** andrekim159@gmail.com  
**Versão atual:** 2.0 (refatoração completa, Junho 2026)

---

## O que é este projeto

Sistema de análise de investimentos em Python que combina notícias em tempo real com dados financeiros históricos para gerar uma decisão de investimento (Comprar / Manter / Vender) usando LLMs.

Fluxo resumido:
1. Coleta notícias sobre uma empresa via NewsAPI (título + descrição, idioma inglês)
2. Envia todas as notícias ao LLM em paralelo (asyncio) para análise de sentimento
3. Coleta histórico de 30 dias da ação + benchmark S&P500 via yfinance
4. Calcula 3 indicadores: tendência (regressão linear), alfa vs S&P500, volume
5. Motor de decisão com pesos combina sentimento (45%) + tendência (35%) + alfa (20%)
6. Gera relatório HTML + JSON + CSV + log persistente

---

## Arquivos do projeto

```
28-Cap17/
├── projeto10-gpt.py          # versão original simples — GPT-4o (OpenAI)
├── projeto10-llama.py        # versão original simples — Llama 3.2 (Ollama)
├── projeto10-gpt-v2.py       # versão refatorada completa — GPT-4o-mini (USAR ESTA)
├── projeto10-llama-fixed.py  # versão refatorada completa — Llama 3.2 (USAR ESTA)
├── requirements.txt          # dependências Python
├── LEIAME.txt                # instruções de setup em português
├── CLAUDE.md                 # este arquivo
├── outputs/                  # gerado em runtime — relatórios HTML, JSON, CSV
└── logs/                     # gerado em runtime — investment_robot.log (append)
```

**Versões a usar:** `projeto10-gpt-v2.py` (GPT) e `projeto10-llama-fixed.py` (Llama) são as versões completas com pipeline assíncrono, indicadores financeiros e geração de relatório HTML.

---

## Setup do ambiente

```bash
# Criar ambiente conda com Python 3.12
conda create --name dsaengfinp10 python=3.12
conda activate dsaengfinp10

# Instalar dependências
conda install pip
pip install -r requirements.txt

# Dependências adicionais da v2 (ainda não no requirements.txt)
pip install scipy jinja2 plotly
```

### Arquivo `.env` (nunca commitar)

Criar `.env` na raiz do projeto:

```dotenv
OPENAI_API_KEY=sk-...          # só para projeto10-gpt-v2.py
NEWSAPI_KEY=sua-chave-aqui     # obrigatório para ambos
```

Obter chave NewsAPI em: newsapi.org (plano gratuito funciona)

### Para a versão Llama (local, sem custo)

```bash
# Instalar Ollama: ollama.com/download
ollama serve          # iniciar servidor
ollama pull llama3.2  # baixar modelo (~4GB, só na primeira vez)
```

---

## Como executar

```bash
# Versão GPT (OpenAI) — requer OPENAI_API_KEY + NEWSAPI_KEY
python projeto10-gpt-v2.py

# Versão Llama (local, gratuita, privada) — requer apenas NEWSAPI_KEY + Ollama rodando
python projeto10-llama-fixed.py
```

Ambos exibem menu interativo pedindo ticker (ex: AAPL) e nome da empresa (ex: Apple).  
Validação automática do ticker via yfinance antes de prosseguir.  
**Apenas empresas NYSE/NASDAQ** — NewsAPI retorna notícias ricas em inglês para essas.

---

## Pipeline técnico detalhado

```
[Menu Interativo] → ticker validado via yfinance.fast_info.last_price
        │
        ▼
[Coleta de Notícias] — NewsAPI: language='en', sort_by='relevancy', page_size=20
        │  Filtra artigos com [Removed] no título
        ▼
[Análise de Sentimento Assíncrona] — asyncio.gather + Semaphore
        │  GPT: MAX_CONCURRENT=5 | Llama: MAX_CONCURRENT=3
        │  Prompt retorna: "positivo - lucros recordes" (categoria - justificativa)
        │  Fallback: neutro se LLM falhar ou resposta não parseável
        ▼
[Dados Financeiros] — yfinance: 30 dias Close+Volume + ^GSPC (S&P500)
        ▼
[Indicadores]
        │  Tendência: scipy.stats.linregress sobre Close 30d → slope % / dia
        │  Volume: volume último dia vs média 30d → sinal alto/normal/baixo
        │  Alfa: retorno ação 30d - retorno S&P500 30d
        ▼
[Motor de Decisão] — pontuação composta com pesos
        │  score = (avg_sentiment * 0.45) + (sign(slope) * 0.35) + (sign(alfa) * 0.20)
        │  volume alto → score * 1.15 | volume baixo → score * 0.85
        │  > +0.25 → COMPRAR | < -0.25 → VENDER | entre → MANTER
        ▼
[Outputs]
        │  outputs/relatorio_{TICKER}_{TIMESTAMP}.html  ← relatório com gráficos
        │  outputs/dados_{TICKER}_{TIMESTAMP}.json
        │  outputs/dados_{TICKER}_{TIMESTAMP}.csv
        └  logs/investment_robot.log  ← append, nunca sobrescreve
```

---

## Decisões de design importantes

- **Scope: só NYSE/NASDAQ** — NewsAPI free tier retorna poucas notícias em português para empresas brasileiras (PETR4, VALE3). Escolha intencional para garantir qualidade de sentimento.
- **asyncio para sentimento** — todas as notícias são analisadas em paralelo para reduzir tempo de ~60s para ~8s (GPT) ou ~90s para ~40s (Llama).
- **Semaphore** — limita chamadas simultâneas ao LLM para evitar rate limit (5 para GPT, 3 para Llama local).
- **Regressão linear** — calcula tendência de 30 dias como slope % por dia, mais robusto que simplesmente comparar primeiro com último preço.
- **Pesos da decisão** — sentimento (45%) pesa mais que tendência (35%) e alfa (20%) porque o LLM analisa contexto qualitativo que dados históricos não capturam.
- **Fallback neutro** — se LLM falhar em qualquer notícia, classifica como neutro e continua sem quebrar o pipeline.

---

## Variáveis de configuração (no topo dos scripts v2)

| Variável | Padrão | Descrição |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` / `llama3.2` | Modelo usado |
| `MAX_CONCURRENT` | 5 (GPT) / 3 (Llama) | Chamadas simultâneas ao LLM |
| `NEWS_PAGE_SIZE` | 20 | Artigos coletados da NewsAPI |
| `HISTORY_PERIOD` | `1mo` | Período histórico de ações |
| `BENCHMARK_TICKER` | `^GSPC` | Índice de referência (S&P500) |
| `BUY_THRESHOLD` | 0.25 | Limiar para decisão COMPRAR |
| `SELL_THRESHOLD` | -0.25 | Limiar para decisão VENDER |
| `OLLAMA_TIMEOUT` | 120 | Timeout Llama em segundos |

---

## Diferenças entre versões GPT e Llama

| Aspecto | `projeto10-gpt-v2.py` | `projeto10-llama-fixed.py` |
|---|---|---|
| LLM | GPT-4o-mini (OpenAI) | Llama 3.2 (Ollama local) |
| Custo | ~$0.01–0.05 por análise | Gratuito |
| Velocidade | ~5–10s | ~30–90s (depende do CPU) |
| Qualidade | Alta e consistente | Boa, pode variar |
| Privacidade | Dados vão para OpenAI | 100% local |
| Credenciais | OPENAI_API_KEY + NEWSAPI_KEY | Apenas NEWSAPI_KEY |
| Pré-requisito extra | Nenhum | Ollama instalado + modelo baixado |

---

## Estrutura dos outputs

### Relatório HTML (`outputs/relatorio_AAPL_*.html`)
- Decisão em destaque com badge colorido + % confiança
- Gráfico: Preço 30 dias + linha de tendência
- Gráfico: Volume 30 dias
- Gráfico: Pizza de sentimentos (positivo/neutro/negativo)
- Tabela de notícias com sentimento e justificativa do LLM
- Aviso legal obrigatório no rodapé

### JSON (`outputs/dados_AAPL_*.json`)
```json
{
  "metadata": {"ticker": "AAPL", "empresa": "Apple", "timestamp": "...", "llm_utilizado": "gpt-4o-mini"},
  "decisao": {"decisao": "COMPRAR", "pontuacao_composta": 0.41, "confianca_pct": 78.5, "fatores": {}},
  "noticias": [{"titulo": "...", "sentimento": "positivo", "justificativa": "...", "score": 1}],
  "indicadores_financeiros": {"tendencia": {}, "volume": {}, "alfa_sp500": {}},
  "precos_30d": [{"data": "2026-05-06", "close": 182.50, "volume": 55000000}]
}
```

---

## Aviso legal

Este projeto é educacional. Não constitui recomendação financeira. Não tome decisões reais de investimento com base neste robô.

---

## Roadmap futuro (fora do escopo atual)

- [ ] Suporte a empresas brasileiras (B3) via RSS Infomoney/Valor Econômico
- [ ] Agendamento automático (cron job) — análise diária
- [ ] Alertas por e-mail/Telegram quando decisão mudar
- [ ] Dashboard web em tempo real (Streamlit)
- [ ] Backtesting: simular performance histórica das decisões
- [ ] Múltiplas empresas em paralelo com relatório comparativo
- [ ] Integração com corretoras via API (ex: Alpaca) para ordens automáticas

---

## Contexto do curso

- **Curso:** DSA — Engenharia Financeira com IA
- **Projeto 10** do curso, foco em LLMs para análise quantitativa
- Versão original (simples) em `projeto10-gpt.py` e `projeto10-llama.py` — usadas nas aulas
- Versão refatorada (v2) desenvolvida em sessões com Claude Code — pipeline completo com async, indicadores financeiros, HTML report
- SPEC.md completo está no Google Drive: pasta "Investimentos robo" (ID: `1yKxgaqMK91hfOWeKz-eOzUrKQr4t_q1M`)
