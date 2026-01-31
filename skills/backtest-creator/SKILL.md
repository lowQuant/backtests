---
name: backtest-creator
description: "Create fast vectorized pandas backtests with data from ib_async, yfinance, or Alpaca; use when a user asks to create or research a trading strategy/backtest, retrieve market data, compute performance metrics (profit factor from per-position daily log returns, Sharpe, max drawdown, win rate, avg win/loss %, CAGR, volatility), and deliver markdown reports, equity-curve vs SPY plots, and per-trade CSVs."
---

# Backtest Creator

Build fast, vectorized pandas backtests first, then offer event-driven evolution if requested.

## Quick workflow

1. Confirm strategy intent and assumptions (signal definition, universe, frequency, execution, costs, slippage, rebalancing).
2. Locate or implement data retrieval (prefer existing utilities). See [references/data-sources.md](references/data-sources.md).
3. Build a vectorized backtest in pandas with explicit position sizing and daily returns.
4. Compute metrics and equity curve. See [references/metrics.md](references/metrics.md).
5. Produce outputs:
   - Markdown report summary
   - Equity curve vs SPY plot
   - Per-trade CSV

## Data and credentials

- Never read `.env` files. Ask the user for the key names to use when credentials are required.
- If credentials are provided out-of-band, accept them explicitly and document expected env var names.
- If data is provided by the user (local files or existing tools), prefer those paths/tools.

## Backtest design guidelines

- Default to vectorized pandas for fast iteration.
- Explicitly model:
  - Position sizing and leverage
  - Entry/exit rules
  - Transaction costs and slippage (even if zero)
  - Trading calendar and missing data handling
- Avoid lookahead bias (shift signals vs. returns).
- Use daily log returns for position-day contributions, especially for profit factor.

## Outputs

- Markdown report: strategy description, assumptions, summary metrics, and brief interpretation.
- Plot: equity curve vs SPY benchmark (use SPY when available; otherwise ask for benchmark).
- Per-trade CSV: include entry/exit timestamps, size, gross/net PnL, return %, and holding period.

If a notebook is requested, generate one alongside the markdown report.
