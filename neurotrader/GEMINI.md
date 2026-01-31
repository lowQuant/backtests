# Neurotrader Project Context

## Project Overview

This workspace contains a collection of Python-based quantitative trading research and automation tools. It appears to be organized into three distinct but related modules focusing on algorithmic trading strategies, technical analysis automation, and statistical validation methods.

### Modules

1.  **`mcpt/` (Monte Carlo Permutation Tests)**
    *   **Purpose:**  Implements tools for statistically validating trading strategies using Monte Carlo permutation tests.
    *   **Key Components:**
        *   `insample_tree_mcpt.py`: Performs in-sample permutation testing on a tree-based strategy.
        *   `bar_permute.py`:  Likely contains logic for shuffling/permuting price bars or returns.
        *   `tree_strat.py`: Defines the tree-based trading strategy being tested.
        *   `donchian.py`: Donchian channel implementation.

2.  **`TechnicalAnalysisAutomation/`**
    *   **Purpose:** A library of functions to automate the detection of classic technical analysis patterns.
    *   **Key Components:**
        *   `directional_change.py`: Implements a directional change algorithm (ZigZag-like) to identify tops and bottoms.
        *   `flags_pennants.py`, `head_shoulders.py`: Detectors for specific chart patterns.
        *   `trendline_automation.py`:  Automated trendline drawing logic.
    *   **Notes:**  Requires specific library versions (see Development Conventions).

3.  **`TrendlineBreakoutMetaLabel/`**
    *   **Purpose:**  Research into trendline breakout strategies, likely incorporating "meta-labeling" (using ML to filter trade signals).
    *   **Key Components:**
        *   `trendline_breakout.py`: Basic implementation of a trendline breakout strategy.
        *   `trendline_break_dataset.py`: Generates datasets (features/labels) for machine learning models to predict breakout success (meta-labeling).
        *   `in_sample_test.py`:  Backtesting script.

## Building and Running

The project consists primarily of standalone Python scripts and research notebooks (converted to `.py`). There is no central build system. Scripts are typically run directly from their respective directories.

**Prerequisites:**
*   Python 3.x
*   Virtual environment recommended.

**Common Dependencies:**
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `mplfinance`
*   `tqdm`
*   `pandas_ta` (specifically for the Trendline module)
*   `pyclustering`

**Setup & Installation:**
There is no global `requirements.txt`. You may need to install dependencies manually.

*   **Critical Note for `TechnicalAnalysisAutomation`:**
    The `pyclustering` library may require an older version of `numpy`.
    ```bash
    pip install numpy==1.23.1
    ```

**Running a Script (Example):**

To run the trendline breakout backtest:
```bash
cd TrendlineBreakoutMetaLabel
python trendline_breakout.py
```

To run the permutation test:
```bash
cd mcpt
python insample_tree_mcpt.py
```

## Development Conventions

*   **Data Handling:**  Scripts expect CSV (`.csv`) or Parquet (`.pq`) files (e.g., `BTCUSDT3600.csv`) in the local directory. Ensure data files are present before running.
*   **Style:**  Code is written in a functional/imperative style common in data science scripts.
*   **Visualization:**  Most scripts are set up to generate `matplotlib` or `mplfinance` charts for visual verification of patterns and strategy performance.
*   **Paths:**  Imports are often relative or assume the script is run from its containing directory (e.g., `from trendline_automation import ...`).
