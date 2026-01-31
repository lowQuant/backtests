# Parameter Stability Plot

Use a simple sweep to visualize how profit factor changes across parameter values.

## Goal

Detect fragile parameter choices and narrow peaks that indicate overfitting.

## Method

1. Choose one parameter (or a small grid for two parameters).
2. Run a backtest for each parameter value.
3. Compute profit factor using the same definition as the main backtest.
4. Plot profit factor (y-axis) vs. parameter value (x-axis).

## Red flags

- A single sharp spike with poor performance on adjacent values.
- High variance in profit factor across small parameter changes.
- Best parameter at the edge of the tested range.

## Reporting

- Include the plot in the audit.
- State the number of parameters and total backtests run for selection.
- Note any sensitivity or unstable regions.
