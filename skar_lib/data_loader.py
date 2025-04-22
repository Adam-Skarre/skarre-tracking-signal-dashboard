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
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df_full = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if 'Adj Close' in df_full.columns:
                df = df_full[['Adj Close']].copy()
                df.rename(columns={'Adj Close': ticker}, inplace=True)
            elif 'Close' in df_full.columns:
                df = df_full[['Close']].copy()
                df.rename(columns={'Close': ticker}, inplace=True)
            else:
                raise ValueError(f"No 'Adj Close' or 'Close' column for {ticker}")
            df.to_csv(cache_path)
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
        if not fname.endswith('.csv'):
            continue
        sym = fname.replace('.csv', '')
        if tickers and sym not in tickers:
            continue
        df = pd.read_csv(os.path.join(cache_dir, fname), index_col=0, parse_dates=True)
        # Ensure correct column name
        if 'Adj Close' in df.columns:
            df = df[['Adj Close']].copy()
            df.rename(columns={'Adj Close': sym}, inplace=True)
        elif 'Close' in df.columns and df.columns.tolist() == ['Close']:
            df = df[['Close']].copy()
            df.rename(columns={'Close': sym}, inplace=True)
        data_frames.append(df)

    if not data_frames:
        return pd.DataFrame()

    price_df = pd.concat(data_frames, axis=1)
    price_df.sort_index(inplace=True)
    return price_df
