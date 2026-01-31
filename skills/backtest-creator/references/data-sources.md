# Data Sources and Retrieval

Prefer existing data utilities in the repo before building new ones. When adding new retrieval code, keep it modular and reusable.

## Priority order

1. Existing repo utilities or user-provided datasets
2. ib_async (IBKR) when available
3. yfinance for fast prototyping
4. Alpaca API for brokerage data

## Credentials and keys

- Never read `.env` files.
- Ask the user for the key names to use (e.g., `ALPACA_API_KEY`, `ALPACA_API_SECRET`).
- If the user provides values directly, use them explicitly and avoid storing them in files.

## ib_async (IBKR)

- Use when the user expects IBKR data access.
- Confirm contract details (symbol, exchange, currency, primary exchange).
- Be explicit about bar size and duration.

## yfinance

- Use for rapid prototyping.
- Confirm interval and date range.
- Note survivorship bias and data quirks when relevant.

## Alpaca

- Use when higher-quality brokerage data is needed.
- Confirm endpoint (paper vs live) and account type.
- Ensure rate limits are respected.
