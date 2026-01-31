import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import arcticdb as adb
import dotenv


def load_arctic():
    dotenv.load_dotenv()

    bucket_name = os.getenv("BUCKET_NAME")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")

    if not bucket_name or not aws_access_key_id or not aws_secret_access_key or not aws_region:
        raise RuntimeError("Missing ArcticDB S3 environment variables.")

    conn_str = (
        f"s3://s3.{aws_region}.amazonaws.com:{bucket_name}"
        f"?region={aws_region}&access={aws_access_key_id}&secret={aws_secret_access_key}"
    )
    return adb.Arctic(conn_str)


def consecutive_negatives(flags: pd.Series) -> np.ndarray:
    streak = 0
    out = []
    for is_neg in flags.fillna(False):
        streak = streak + 1 if is_neg else 0
        out.append(streak)
    return np.array(out, dtype=int)


def find_current_signals(all_stocks: pd.DataFrame):
    if not {"Close", "200D_EMA", "1d_log_ret", "Symbol"}.issubset(all_stocks.columns):
        raise ValueError("ALL_STOCKS is missing required columns.")

    all_stocks = all_stocks.sort_index().copy()

    parts = []
    for symbol, sym_df in all_stocks.groupby("Symbol"):
        sym = sym_df.copy()
        sym["is_negative"] = sym["1d_log_ret"] < 0
        sym["consecutive_decline"] = consecutive_negatives(sym["is_negative"])
        sym["signal_4d_decline"] = (
            (sym["Close"] > sym["200D_EMA"]) & (sym["consecutive_decline"] == 4)
        )
        parts.append(sym)

    if not parts:
        return [], pd.DataFrame()

    df = pd.concat(parts).sort_index()

    last_date = df.index.max()
    signal_stocks = df[(df["signal_4d_decline"]) & (df.index == last_date)]
    symbols = sorted(signal_stocks["Symbol"].unique().tolist())
    return symbols, signal_stocks


def download_history(symbols):
    if not symbols:
        return pd.DataFrame()

    hist = yf.download(
        symbols,
        period="5y",
        group_by="ticker",
        auto_adjust=False,
        multi_level_index=True,
        progress=False,
    )
    if hist.empty:
        return pd.DataFrame()

    hist_df = (
        hist.stack(level=0)
        .rename_axis(["Date", "Symbol"])
        .reset_index(level=1)
        .sort_index(kind="stable")
    )
    return hist_df


def backtest_signal(hist_df: pd.DataFrame) -> pd.DataFrame:
    if hist_df.empty:
        return pd.DataFrame()

    if not {"Open", "Close", "Symbol"}.issubset(hist_df.columns):
        raise ValueError("Historical data is missing required columns.")

    records = []

    for symbol, sym_df in hist_df.groupby("Symbol"):
        sym = sym_df.sort_index().copy()

        sym["close_log_ret"] = np.log(sym["Close"]) - np.log(sym["Close"].shift(1))
        sym["intraday_ret"] = np.log(sym["Close"]) - np.log(sym["Open"])
        sym["fwd_intraday_ret"] = sym["intraday_ret"].shift(-1)
        sym["ma_200"] = sym["Close"].rolling(window=200, min_periods=200).mean()

        sym["is_negative"] = sym["close_log_ret"] < 0
        sym["consecutive_decline"] = consecutive_negatives(sym["is_negative"])

        sym["signal"] = (
            (sym["consecutive_decline"] == 4) & (sym["Close"] > sym["ma_200"])
        )

        trades = sym.loc[sym["signal"], "fwd_intraday_ret"].dropna()
        n_trades = int(trades.shape[0])

        if n_trades >= 2:
            mean_ret = trades.mean()
            std_ret = trades.std(ddof=1)
            sharpe = mean_ret / std_ret if std_ret > 0 else np.nan
        elif n_trades == 1:
            mean_ret = trades.iloc[0]
            std_ret = np.nan
            sharpe = np.nan
        else:
            mean_ret = np.nan
            std_ret = np.nan
            sharpe = np.nan

        records.append(
            {
                "Symbol": symbol,
                "n_trades": n_trades,
                "mean_trade_ret": mean_ret,
                "std_trade_ret": std_ret,
                "sharpe_per_trade": sharpe,
            }
        )

    stats_df = pd.DataFrame.from_records(records).set_index("Symbol").sort_values(
        by="sharpe_per_trade", ascending=False
    )
    return stats_df


def main():
    print("Connecting to ArcticDB and loading ALL_STOCKS...")
    ac = load_arctic()
    us_equities_lib = ac.get_library("us_equities")
    all_stocks = us_equities_lib.read("ALL_STOCKS").data

    print("Finding current uptrend + 4-day decline signals on the latest trading day...")
    current_symbols, signal_stocks = find_current_signals(all_stocks)
    last_date = signal_stocks.index.max() if not signal_stocks.empty else None
    print(f"Last date in ALL_STOCKS: {last_date}")
    print(f"Current symbols with signal on last date: {len(current_symbols)}")

    if not current_symbols:
        print("No current signals found on the latest trading day. Exiting.")
        return

    print("Downloading 5Y historical data for signal symbols from yfinance...")
    hist_df = download_history(current_symbols)
    if hist_df.empty:
        print("Historical data download returned empty. Exiting.")
        return

    print("Running historical backtest of the signal...")
    stats_df = backtest_signal(hist_df)
    if stats_df.empty:
        print("No historical trades found for the signal. Exiting.")
        return

    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "lt_uptrend_4d_decline_sharpe_by_symbol.csv"
    stats_df.to_csv(out_path)
    print(f"Saved per-symbol stats to: {out_path}")

    qualified = stats_df[stats_df["sharpe_per_trade"] > 1].index.tolist()
    print("\nSymbols with historical Sharpe (per-trade) > 1:")
    if qualified:
        for s in qualified:
            print(s)
    else:
        print("None")


if __name__ == "__main__":
    main()
