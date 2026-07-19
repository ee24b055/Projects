import pandas as pd
import numpy as np

def create_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Feature 1: Asset daily return percentage
    df['Return'] = df['Close'].pct_change()
    
    # Feature 2: Trend indicator (Ratio of Short-term vs Long-term moving averages)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['SMA_Ratio'] = df['SMA_10'] / df['SMA_30']
    
    # Feature 3: Momentum Indicator (Relative Strength Index - RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)  # prevent ZeroDivisionError
    df['RSI'] = 100 - (100 / (1 + rs)) # >70: overbought , <30: less- value, price may rise
    
    # Target: Predict if tomorrow's price is higher than today's close (1 = Up, 0 = Down)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    return df.dropna()


## other feature : volatility   df['Volatility'] = df['Return'].rolling(window=10).std()
