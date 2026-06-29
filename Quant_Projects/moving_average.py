import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
data = yf.download("SPY", start="2015-01-01", end="2024-01-01")
data = data[['Close']].copy()
data['MA_short'] = data['Close'].rolling(50).mean()
data['MA_long'] = data['Close'].rolling(200).mean()

data['signal'] = 0
data.loc[data['MA_short'] > data['MA_long'], 'signal'] = 1

data['daily_return'] = data['Close'].pct_change()
data['strategy_return'] = data['signal'].shift(1) * data['daily_return']
data.dropna(inplace=True)


annualised_return = data['strategy_return'].mean() * 252
annualised_vol = data['strategy_return'].std() * np.sqrt(252)
sharpe_ratio = annualised_return / annualised_vol
cumulative = (1 + data['strategy_return']).cumprod()
rolling_max = cumulative.cummax()
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = drawdown.min()
print(f"Annualised Return: {annualised_return:.2%}")
print(f"Annualised Volatility: {annualised_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
data['Close'].plot(ax=ax1, label='SPY Price')
data['MA_short'].plot(ax=ax1, label='50-day MA')
data['MA_long'].plot(ax=ax1, label='200-day MA')
ax1.legend()
ax1.set_title('Price and Moving Averages')
(1 + data['strategy_return']).cumprod().plot(ax=ax2, label='Strategy')
(1 + data['daily_return']).cumprod().plot(ax=ax2, label='Buy and Hold')
ax2.legend()
ax2.set_title('Equity Curves')
drawdown.plot(ax=ax3, color='red')
ax3.set_title('Strategy Drawdown')
ax3.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
plt.tight_layout()
plt.show()