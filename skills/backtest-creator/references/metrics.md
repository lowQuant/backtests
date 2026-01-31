# Metrics Reference

Use these definitions unless the user specifies alternatives.

## Returns

- **Position-day log return**: `log(1 + position_return)` for each position on each day.
- **Portfolio daily return**: sum of position-day returns weighted by allocation; use log returns if composing.

## Profit Factor (position-day)

Compute from all position-day log returns (not per-trade totals):

```
profit_factor = sum(positive_position_day_log_returns) / abs(sum(negative_position_day_log_returns))
```

## Sharpe Ratio

```
sharpe = mean(daily_returns) / std(daily_returns) * sqrt(annualization_factor)
```

Use `annualization_factor = 252` for daily data unless specified.

## Max Drawdown

- Compute from the equity curve using rolling peak-to-trough decline.

## Win Rate

```
win_rate = count(trades_with_positive_net_return) / total_trades
```

## Average Win % / Average Loss %

- Average % return of winning trades vs losing trades.

## CAGR

```
cagr = (equity_end / equity_start) ** (1 / years) - 1
```

## Volatility

```
volatility = std(daily_returns) * sqrt(annualization_factor)
```
