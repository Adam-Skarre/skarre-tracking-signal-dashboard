import os
import pandas as pd
import yfinance as yf


def download_price_data(tickers, start_date, end_date, cache_dir='data'):
    """
    Download adjusted close price data for given tickers from Yahoo Finance.
    Caches results locally to avoid repeated downloads.
    Returns a DataFrame with Date index and tickers as columns.
    """
    os.makedirs(cache_dir, exist_ok=True)
    data_frames = []

    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker}.csv")
        if os.path.isfile(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df_full = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if 'Adj Close' in df_full.columns:
                df_ticker = df_full[['Adj Close']].copy()
                df_ticker.rename(columns={'Adj Close': ticker}, inplace=True)
            elif 'Close' in df_full.columns:
                df_ticker = df_full[['Close']].copy()
                df_ticker.rename(columns={'Close': ticker}, inplace=True)
            else:
                raise ValueError(f"No 'Adj Close' or 'Close' column for {ticker}")
            df_ticker.to_csv(cache_path)
            df = df_ticker
        data_frames.append(df)

    price_df = pd.concat(data_frames, axis=1)
    price_df.sort_index(inplace=True)
    return price_df


def load_cached_data(cache_dir='data', tickers=None):
    """
    Load cached price CSVs from cache_dir. If tickers is provided, load those only.
    Returns a DataFrame with Date index and tickers as columns.
    """
    if not os.path.isdir(cache_dir):
        return pd.DataFrame()

    data_frames = []
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.lower().endswith('.csv'):
            continue
        symbol = fname[:-4]
        if tickers and symbol not in tickers:
            continue
        df = pd.read_csv(os.path.join(cache_dir, fname), index_col=0, parse_dates=True)
        if 'Adj Close' in df.columns:
            df = df[['Adj Close']].copy()
            df.rename(columns={'Adj Close': symbol}, inplace=True)
        elif 'Close' in df.columns:
            df = df[['Close']].copy()
            df.rename(columns={'Close': symbol}, inplace=True)
        data_frames.append(df)

    if not data_frames:
        return pd.DataFrame()

    price_df = pd.concat(data_frames, axis=1)
    price_df.sort_index(inplace=True)
    return price_df
