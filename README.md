This is a directory for backtests.

There is a utils folder that contains some useful functions for downloading data.

All backtests should be in a separate folder.

Currently we are working on a backtest for an intraday breakout strategy on QQQ.

### Environment Setup
It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create virtual environment (using python 3.10+)
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Downloading
The utility script `utils/ib_intraday_downloader.py` can be used to download historical data from Interactive Brokers.

**Usage:**
```bash
# Ensure venv is active or use full path
./venv/bin/python3 -m utils.ib_intraday_downloader --symbol SYMBOL [--start START_DATE] [--end END_DATE] [--interval INTERVAL]
```

**Example:**
Download QQQ 1-minute data from 2020-01-01 to today:
```bash
./venv/bin/python3 -m utils.ib_intraday_downloader --symbol QQQ --start 2020-01-01 --end today
```

The data will be saved as a Parquet file in the `data/` directory, named `SYMBOL_START_END.parquet`.