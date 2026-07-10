import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request
import zipfile
import io

def download_fama_french_factors():
    """Downloads daily Fama-French 3-Factor model datasets from Ken French's data library."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, skiprows=4, index_col=0, parse_dates=False)
                
    df = df.dropna()
    df.index = pd.to_datetime(df.index, format='%Y%m%d', errors='coerce')
    df = df.dropna()
    return df / 100.0

# def get_aligned_data(tickers, benchmark, start_date, end_date):
#     """Downloads asset data via yfinance and merges it chronologically with risk factor matrices."""
#     print(f"Downloading historical data via yfinance...")
#     all_assets = tickers + [benchmark]
#     data = yf.download(all_assets, start=start_date, end=end_date, progress=False)['Adj Close']
    
#     returns = np.log(data / data.shift(1)).dropna()
#     ff_factors = download_fama_french_factors()
    
#     common_idx = returns.index.intersection(ff_factors.index)
#     return returns.loc[common_idx, tickers], returns.loc[common_idx, benchmark], ff_factors.loc[common_idx]
def get_aligned_data(tickers, benchmark, start_date, end_date):
    """Downloads asset data via yfinance and merges it chronologically with risk factor matrices."""
    print(f"Downloading historical data via yfinance...")
    all_assets = tickers + [benchmark]
    
    # 1. Force auto_adjust=False to preserve 'Adj Close' across standard yfinance versions
    data = yf.download(all_assets, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # 2. Version-agnostic fallback checking for the column layout
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        raise KeyError("Could not extract pricing data column matrix from yfinance.")
    
    returns = np.log(prices / prices.shift(1)).dropna()
    ff_factors = download_fama_french_factors()
    
    common_idx = returns.index.intersection(ff_factors.index)
    return returns.loc[common_idx, tickers], returns.loc[common_idx, benchmark], ff_factors.loc[common_idx]