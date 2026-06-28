import yfinance as yf
import pandas as pd

def fetch_historical_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"[-] Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Clean up column names in case yfinance passes MultiIndex columns
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df