# Backtest Bias and Stability Checklist

Use this checklist to identify common flaws and risks.

## Look-ahead bias (highest priority)

- Signals use future prices, fundamentals, or labels.
- Indicators computed with data not available at decision time.
- Execution price uses same-bar close without proper delay.
- Rolling windows include the current bar when signal should be based on prior data.

## Data leakage

- Feature engineering uses full-sample statistics (mean, std, scaling) without proper training split.
- Label leakage from future returns or target construction.
- Data preprocessing fit on the full dataset.

## Survivorship and selection bias

- Universe defined by current constituents only.
- Delisted assets omitted.
- Results depend on a post-hoc chosen universe.

## Transaction costs and execution

- Costs, slippage, and liquidity ignored or unrealistic.
- Fills assumed at prices not achievable given volume/spread.
- Latency and order timing mismatches with data frequency.

## Overfitting and curve fitting

- Too many parameters relative to data length.
- Performance collapses out-of-sample or on regime shifts.
- Parameter sensitivity is high; performance is driven by a narrow window.
- Multiple testing risk: many backtests/parameter sweeps run to select the best result.
- Report the number of parameters and the number of backtests run to reach the final result.

## Stability and robustness

- Results depend on a small number of trades or a short period.
- High turnover with fragile edge.
- Metrics are unstable across assets or time slices.

## Benchmarking and attribution

- Performance primarily from market beta, not alpha.
- Strategy underperforms after risk adjustment.

## Implementation pitfalls

- Missing data handling introduces bias (forward-fill of future values).
- Corporate actions not handled.
- Timezone and session alignment errors.
