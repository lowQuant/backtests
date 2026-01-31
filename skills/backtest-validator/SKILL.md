---
name: backtest-validator
description: "Review backtest code and results for flaws, biases, and instability; use when asked to validate or audit a backtest, detect look-ahead bias, data leakage, survivorship bias, curve fitting, or overfitting, and produce a structured markdown audit with risks, tests, and remediation steps."
---

# Backtest Validator

Audit backtest implementations and results for biases and instability, with special focus on look-ahead bias.

## Example triggers

- "Validate this backtest for bias and overfitting risk."
- "Audit my strategy code for look-ahead bias and data leakage."
- "Review these results and tell me if they are stable or overfit."

## Workflow

1. Identify the backtest scope: data sources, frequency, assets, signals, execution assumptions, and evaluation period.
2. Read code and outputs; trace data flow from raw inputs to signals and trades.
3. Run the bias checklist in [references/checklist.md](references/checklist.md).
4. Assess stability and overfitting risk; propose robustness tests (include parameter stability plots).
5. Produce a concise markdown audit with findings, severity, evidence, and fixes.

## Output format

- **Findings**: ordered by severity, with file/line references when available.
- **Tests**: robustness checks to run (e.g., walk-forward, parameter sweeps, different regimes). Include parameter stability plots from [references/parameter-stability.md](references/parameter-stability.md).
- **Remediations**: code or methodology changes to reduce bias.
- **Open questions**: data, execution, and assumptions to confirm.

## Special rules

- Treat look-ahead bias as highest severity; confirm all signals are shifted to avoid future leakage.
- Verify that data availability matches execution time (e.g., EOD prices used only after market close).
- Flag unstable metrics, parameter sensitivity, and performance dominated by a small subset of trades.
- If a local MCPT folder exists, review it for permutation testing patterns and reuse when relevant.
