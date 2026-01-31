# Example: ./venv/bin/python3 -m utils.ib_intraday_downloader --symbol TQQQ --mode intraday --start 2020-01-01 --end today
"""
IB historical downloader utilities with rate limiting and progress callbacks.
- Uses ib_async IB client
- Paginates backwards using 20 D chunks for 1-minute bars (RTH=True) to stay under limits
- Reports progress via a callback
"""
from __future__ import annotations

import asyncio
import argparse
import os
from typing import Callable, List, Optional, Tuple
import time
from collections import deque
from datetime import datetime

import pandas as pd
from ib_async import Stock, util

from .ib_connection import connect_to_ib, disconnect_from_ib
from core.log_manager import add_log


class RateLimiter:
    """Rate limiter for IB historical data requests.

    IB Rules (approx, enforced conservatively):
    - Max 6 identical requests per 2 seconds (we cap to 5)
    - Max 60 requests per 10 minutes (we cap to 59)
    - Add minimum spacing between requests
    """

    def __init__(self) -> None:
        self.requests_2s: deque[float] = deque()
        self.requests_10m: deque[float] = deque()
        self.last_request_time: float = 0.0

    async def wait(self) -> None:
        now = time.time()
        # Evict old timestamps
        while self.requests_2s and now - self.requests_2s[0] > 2.0:
            self.requests_2s.popleft()
        while self.requests_10m and now - self.requests_10m[0] > 600.0:
            self.requests_10m.popleft()
        # If at limit, sleep until a slot frees up
        if len(self.requests_2s) >= 5:
            wait_time = 2.1 - (now - self.requests_2s[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = time.time()
        if len(self.requests_10m) >= 59:
            wait_time = 600.1 - (now - self.requests_10m[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                now = time.time()
        # Ensure small spacing between requests
        if now - self.last_request_time < 0.4:
            await asyncio.sleep(0.4 - (now - self.last_request_time))
            now = time.time()
        # Record
        self.requests_2s.append(now)
        self.requests_10m.append(now)
        self.last_request_time = now


def _interval_to_barsize(interval: str) -> str:
    it = (interval or "minute").lower()
    if it in {"minute", "1m", "1min"}:
        return "1 min"
    if it in {"hour", "hourly", "60m", "1h"}:
        return "1 hour"
    if it in {"day", "daily", "1d"}:
        return "1 day"
    return "1 min"


def _default_chunk(interval: str, use_rth: bool) -> str:
    # For 1-min RTH we can safely ask up to about 20 D per page (~8k-10k bars)
    it = (interval or "minute").lower()
    if it in {"minute", "1m", "1min"}:
        return "20 D" if use_rth else "7 D"
    if it in {"hour", "hourly", "60m", "1h"}:
        return "90 D"
    if it in {"day", "daily", "1d"}:
        return "365 D"
    return "20 D"


def _parse_duration_days(duration: str) -> int:
    try:
        parts = duration.strip().split()
        if len(parts) != 2:
            return 20
        val = int(parts[0])
        unit = parts[1].upper()
        if unit.startswith('D'):
            return val
        if unit.startswith('W'):
            return val * 7
        if unit.startswith('M'):
            return val * 30
        if unit.startswith('Y'):
            return val * 365
        return val
    except Exception:
        return 20


def _progress_percent(start_ts: pd.Timestamp, end_ts: pd.Timestamp, reached_ts: pd.Timestamp) -> float:
    try:
        total = (end_ts - start_ts).total_seconds()
        covered = (end_ts - reached_ts).total_seconds()
        if total <= 0:
            return 100.0
        pct = max(0.0, min(100.0, 100.0 * covered / total))
        return pct
    except Exception:
        return 0.0


async def download_ib_historical_paginated(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    *,
    use_rth: bool = True,
    what_to_show: str = "TRADES",
    chunk: Optional[str] = None,
    client_id: int = 9999,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """Download historical bars by paging backwards from end->start.

    - Returns a DataFrame with index as timestamp and columns: open, high, low, close, volume
    - The function does not write to ArcticDB; the caller decides.
    - Emits progress via progress_cb(percentage, message) if provided.
    """
    # Normalize date bounds
    full_lookback = isinstance(start_date, str) and start_date.strip().lower() in {"max", "all", "full"}
    start_ts = pd.to_datetime("1900-01-01") if full_lookback else pd.to_datetime(start_date)
    # Compute end datetime in US/Eastern for IB requests and a naive clamp for slicing
    if end_date and end_date.strip().lower() != "today":
        tmp_end = pd.to_datetime(end_date)
        # If time not provided (00:00:00), use end-of-day Eastern
        if tmp_end.hour == 0 and tmp_end.minute == 0 and tmp_end.second == 0:
            end_eastern = (pd.Timestamp(tmp_end.date(), tz="US/Eastern") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        else:
            # Assume provided time is Eastern
            end_eastern = pd.Timestamp(tmp_end.to_pydatetime(), tz="US/Eastern") if tmp_end.tzinfo is None else tmp_end.tz_convert("US/Eastern")
    else:
        # today: use current Eastern time
        end_eastern = pd.Timestamp.now(tz="US/Eastern")
    end_ts = end_eastern.tz_convert(None)  # naive wall-clock for slicing

    ib = await connect_to_ib(client_id=client_id, symbol=symbol.upper())
    if not ib or not ib.isConnected():
        raise RuntimeError("Failed to connect to IB")

    limiter = RateLimiter()
    try:
        contract = Stock(symbol.upper(), "SMART", "USD")
        await ib.qualifyContractsAsync(contract)
        barsize = _interval_to_barsize(interval)
        duration = chunk or _default_chunk(interval, use_rth)
        chunk_days = _parse_duration_days(duration)

        dt = end_eastern.strftime("%Y%m%d %H:%M:%S US/Eastern")
        pages = 0
        frames: List[pd.DataFrame] = []
        prev_earliest_ts = end_ts  # Initialize with end timestamp

        # Determine a practical pages limit to avoid accidental infinite loops
        if not full_lookback:
            days_total = max(1, (end_ts.date() - start_ts.date()).days + 1)
            pages_limit = max(1, (days_total // chunk_days) + 3)
        else:
            pages_limit = 1000

        while pages < pages_limit:
            await limiter.wait()
            pages += 1
            if progress_cb:
                # For the first page, show 0% (we haven't fetched anything yet)
                if pages == 1:
                    progress_cb(0.0, f"Fetching page {pages} end={dt or 'now'} dur={duration}")
                else:
                    # Use the previous page's earliest timestamp for progress
                    progress_cb(_progress_percent(start_ts, end_ts, prev_earliest_ts), f"Fetching page {pages} end={dt or 'now'} dur={duration}")
            try:
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=dt or "",
                    durationStr=duration,
                    barSizeSetting=barsize,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    keepUpToDate=False,
                    formatDate=1,
                )
            except Exception as e:
                # retry once
                await asyncio.sleep(2.0)
                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime=dt or "",
                    durationStr=duration,
                    barSizeSetting=barsize,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    keepUpToDate=False,
                    formatDate=1,
                )

            if not bars:
                break

            # Convert and append
            part = util.df(bars)
            # util.df returns 'date', 'open','high','low','close','volume', ... and index numeric; set datetime index in US/Eastern (tz-naive)
            if "date" in part.columns:
                dt_series = pd.to_datetime(part["date"])  # type: ignore
                try:
                    if getattr(dt_series.dt, 'tz', None) is not None:
                        dt_series = dt_series.dt.tz_convert('US/Eastern').dt.tz_localize(None)
                    else:
                        # Assume the naive timestamps are already Eastern
                        dt_series = dt_series
                except Exception:
                    # Fallback: keep as parsed
                    pass
                part["timestamp"] = dt_series
                part = part.set_index("timestamp")
            # keep only ohlcv
            keep = [c for c in ["open", "high", "low", "close", "volume"] if c in part.columns]
            part = part[keep].sort_index()
            frames.append(part)

            # Determine next end time from earliest bar object directly to avoid tz drift
            earliest_bar_dt = getattr(bars[0], 'date', None) or getattr(bars[0], 'time', None)
            if earliest_bar_dt is None:
                break
            earliest_ts = pd.to_datetime(earliest_bar_dt)
            # Update progress
            if progress_cb:
                progress_cb(_progress_percent(start_ts, end_ts, earliest_ts.tz_convert(None) if getattr(earliest_ts, 'tzinfo', None) else earliest_ts), f"Fetched page {pages} rows={len(part)} earliest={earliest_ts}")
            # Update prev_earliest_ts for next iteration
            prev_earliest_ts = earliest_ts.tz_convert(None) if getattr(earliest_ts, 'tzinfo', None) else earliest_ts
            # Stop if we've covered past start_ts (unless full-lookback requested)
            earliest_naive = earliest_ts.tz_convert(None) if getattr(earliest_ts, 'tzinfo', None) else earliest_ts
            if not full_lookback and earliest_naive <= start_ts:
                break
            # Build next dt in US/Eastern and step back 1s to avoid identical end
            if getattr(earliest_ts, 'tzinfo', None) is None:
                # assume US/Eastern for IB historical naive timestamps
                earliest_ts = earliest_ts.tz_localize('US/Eastern')
            ts_eastern = earliest_ts.tz_convert('US/Eastern') - pd.Timedelta(seconds=1)
            dt = ts_eastern.strftime("%Y%m%d %H:%M:%S US/Eastern")

            # Small pacing delay between pages for stability
            await asyncio.sleep(0.8)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()
        # Deduplicate and clamp to range
        df = df[~df.index.duplicated(keep="last")]
        # Clamp to end_ts (start_ts only clamps if not full-lookback)
        df = df.loc[:, :]
        df = df.loc[:end_ts]
        if not full_lookback:
            df = df.loc[start_ts:]
        if progress_cb:
            progress_cb(100.0, f"Completed. Rows={len(df)}")
        return df
    finally:
        await disconnect_from_ib(ib, symbol=symbol)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Download IB historical data to Parquet")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol (e.g., QQQ)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="today", help="End date (YYYY-MM-DD) or 'today'")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["intraday", "daily"],
        default="intraday",
        help="Download mode (intraday=1m bars, daily=1d bars)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="Optional custom bar interval (overrides --mode)",
    )
    parser.add_argument("--client-id", type=int, default=9999, help="IB client ID")
    
    args = parser.parse_args()

    interval = args.interval or ("1m" if args.mode == "intraday" else "1d")

    print(f"Starting {args.mode} download for {args.symbol} from {args.start} to {args.end}...")
    
    def progress(pct: float, msg: str) -> None:
        print(f"[{pct:.1f}%] {msg}")

    try:
        df = await download_ib_historical_paginated(
            symbol=args.symbol,
            interval=interval,
            start_date=args.start,
            end_date=args.end,
            client_id=args.client_id,
            progress_cb=progress
        )

        if not df.empty:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Format filename: SYMBOL_MODE_STARTYEAR-ENDYEAR.parquet
            min_ts = pd.to_datetime(df.index.min())
            max_ts = pd.to_datetime(df.index.max())
            period_suffix = f"{min_ts.year}-{max_ts.year}"
            mode_suffix = args.mode.lower()
            filename = f"data/{args.symbol.upper()}_{mode_suffix}_{period_suffix}.parquet"
            
            df.to_parquet(filename)
            print(f"Data saved to {filename} ({len(df)} rows)")
        else:
            print("No data downloaded.")
    except Exception as e:
        print(f"Error during download: {e}")

if __name__ == "__main__":
    asyncio.run(main())
