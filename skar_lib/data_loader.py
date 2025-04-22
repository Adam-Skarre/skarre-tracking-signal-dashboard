import os
import pandas as pd
import yfinance as yf


def download_price_data(tickers, start_date, end_date, cache_dir='data'):
    """
    Download adjusted close price data for given tickers from Yahoo Finance.
    Caches results locally to avoid repeated downloads.

    Parameters:
    - tickers: list of ticker symbols (e.g. ['SPY', 'QQQ'])
    - start_date: string in 'YYYY-MM-DD'
    - end_date: string in 'YYYY-MM-DD'
    - cache_dir: directory to store cached CSV files

    Returns:
    - DataFrame with Date index and tickers as columns
    """
    os.makedirs(cache_dir, exist_ok=True)
    all_data = []

    for ticker in tickers:
        cache_path = os.path.join(cache_dir, f"{ticker}.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)[['Adj Close']]
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.to_csv(cache_path)
        all_data.append(df[ticker])

    price_df = pd.concat(all_data, axis=1)
    price_df.sort_index(inplace=True)
    return price_df


def load_cached_data(cache_dir='data', tickers=None):
    """
    Load cached price CSVs from cache_dir. If tickers is provided,
    only loads those; otherwise loads all CSV files.

    Returns:
    - DataFrame with Date index and tickers as columns
    """
    files = os.listdir(cache_dir)
    data_frames = []
    for fname in files:
        if fname.endswith('.csv') and (tickers is None or fname.replace('.csv', '') in tickers):
            df = pd.read_csv(os.path.join(cache_dir, fname), index_col=0, parse_dates=True)
            # assume CSV has either a column matching ticker or 'Adj Close'
            if 'Adj Close' in df.columns:
                df = df[['Adj Close']]
                df.rename(columns={'Adj Close': fname.replace('.csv', '')}, inplace=True)
            data_frames.append(df)

    if not data_frames:
        return pd.DataFrame()

    price_df = pd.concat(data_frames, axis=1)
    price_df.sort_index(inplace=True)
    return price_df
