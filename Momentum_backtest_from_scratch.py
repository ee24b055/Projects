#This code downloads monthly sector ETF prices, ranks the sectors by trailing momentum, and each month invests equally in the top 3 performers.

#It then applies transaction costs based on portfolio turnover, compares the strategy with an equal-weight benchmark, and plots the equity curve, drawdown, and rolling returns.

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sector ETFs used for the rotation strategy
TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE"]

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

LOOKBACK_MONTHS = 12
TOP_N = 3
COST_PER_UNIT_TURNOVER = 0.001   # 0.10% transaction cost
RISK_FREE_ANNUAL = 0.02          # 2% annual risk-free rate


def annualized_return(monthly_returns):
    compounded = (1 + monthly_returns).prod()
    n_months = len(monthly_returns)
    if n_months == 0:
        return np.nan
    return compounded ** (12 / n_months) - 1


def annualized_volatility(monthly_returns):
    return monthly_returns.std() * np.sqrt(12)


def sharpe_ratio(monthly_returns, risk_free_annual=0.02):
    rf_monthly = (1 + risk_free_annual) ** (1 / 12) - 1
    excess_returns = monthly_returns - rf_monthly
    vol = excess_returns.std()

    if vol == 0 or np.isnan(vol):
        return np.nan

    return (excess_returns.mean() / vol) * np.sqrt(12)


def max_drawdown(monthly_returns):
    equity_curve = (1 + monthly_returns).cumprod()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1
    return drawdown.min(), drawdown, equity_curve


def compute_momentum_weights(monthly_prices, lookback_months=12, top_n=3):
    momentum_scores = monthly_prices.pct_change(lookback_months)

    weights = pd.DataFrame(
        0.0,
        index=monthly_prices.index,
        columns=monthly_prices.columns
    )

    for date in momentum_scores.index[lookback_months:]:
        scores = momentum_scores.loc[date].dropna()

        if len(scores) < top_n:
            continue

        top_assets = scores.sort_values(ascending=False).head(top_n).index
        weights.loc[date, top_assets] = 1.0 / top_n

    return momentum_scores, weights


def compute_strategy_returns(monthly_returns, weights, cost_per_turnover=0.001):
    shifted_weights = weights.shift(1).fillna(0.0)

    gross_returns = (shifted_weights * monthly_returns).sum(axis=1)

    turnover = shifted_weights.diff().abs().sum(axis=1).fillna(0.0)
    transaction_cost = turnover * cost_per_turnover

    net_returns = gross_returns - transaction_cost

    result = pd.DataFrame({
        "gross_return": gross_returns,
        "turnover": turnover,
        "transaction_cost": transaction_cost,
        "net_return": net_returns
    })

    return result


def summarize_performance(monthly_returns, risk_free_annual=0.02):
    ann_ret = annualized_return(monthly_returns)
    ann_vol = annualized_volatility(monthly_returns)
    sharpe = sharpe_ratio(monthly_returns, risk_free_annual)
    mdd, drawdown, equity_curve = max_drawdown(monthly_returns)

    summary = {
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Maximum Drawdown": mdd
    }

    return summary, equity_curve, drawdown


print("Downloading ETF price data...")

raw = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)

if raw.empty:
    raise ValueError("No data was downloaded. Please check the ticker list or internet connection.")

# yfinance can return multi-index columns when downloading multiple tickers
if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"].copy()
else:
    prices = raw.copy()

prices = prices.dropna(how="all").ffill().dropna()

# Convert daily prices to month-end prices for monthly rebalancing
monthly_prices = prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna(how="all")

# Keep only columns that have usable return data
monthly_returns = monthly_returns.dropna(axis=1, how="all")
monthly_prices = monthly_prices[monthly_returns.columns]

print("Tickers used in the backtest:")
print(list(monthly_returns.columns))

momentum_scores, raw_weights = compute_momentum_weights(
    monthly_prices,
    lookback_months=LOOKBACK_MONTHS,
    top_n=TOP_N
)

strategy_results = compute_strategy_returns(
    monthly_returns=monthly_returns,
    weights=raw_weights,
    cost_per_turnover=COST_PER_UNIT_TURNOVER
)

valid_index = strategy_results["net_return"].notna()
strategy_results = strategy_results.loc[valid_index]
strategy_weights = raw_weights.loc[valid_index]
momentum_scores = momentum_scores.loc[valid_index]
monthly_returns = monthly_returns.loc[valid_index]

# Equal-weight benchmark across all available sector ETFs
benchmark_returns = monthly_returns.mean(axis=1)

strategy_summary, strategy_equity, strategy_drawdown = summarize_performance(
    strategy_results["net_return"],
    risk_free_annual=RISK_FREE_ANNUAL
)

benchmark_summary, benchmark_equity, benchmark_drawdown = summarize_performance(
    benchmark_returns,
    risk_free_annual=RISK_FREE_ANNUAL
)

summary_df = pd.DataFrame({
    "Strategy": strategy_summary,
    "Equal-Weight Benchmark": benchmark_summary
})

avg_turnover = strategy_results["turnover"].mean()
annual_turnover = avg_turnover * 12
total_cost_paid = strategy_results["transaction_cost"].sum()

print("\nPerformance summary")
print(summary_df.round(4))

print("\nStrategy settings")
print(f"Lookback window: {LOOKBACK_MONTHS} months")
print(f"Top assets selected each month: {TOP_N}")
print(f"Transaction cost per unit turnover: {COST_PER_UNIT_TURNOVER:.2%}")
print(f"Average monthly turnover: {avg_turnover:.4f}")
print(f"Estimated annual turnover: {annual_turnover:.4f}")
print(f"Total transaction cost paid: {total_cost_paid:.4f}")

print("\nRecent momentum scores")
print(momentum_scores.tail(8).round(4))

print("\nRecent portfolio weights")
print(strategy_weights.tail(8).round(3))

print("\nRecent strategy return data")
print(strategy_results.tail(8).round(4))

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

strategy_equity.plot(ax=axes[0], label="Cross-Sectional Momentum", linewidth=2)
benchmark_equity.plot(ax=axes[0], label="Equal-Weight Benchmark", linewidth=2, alpha=0.8)
axes[0].set_title("Equity Curve")
axes[0].set_ylabel("Growth of $1")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

strategy_drawdown.plot(ax=axes[1], color="red", linewidth=1.5, label="Strategy Drawdown")
axes[1].fill_between(strategy_drawdown.index, strategy_drawdown.values, 0, color="red", alpha=0.25)
benchmark_drawdown.plot(ax=axes[1], color="gray", linewidth=1.2, alpha=0.7, label="Benchmark Drawdown")
axes[1].set_title("Drawdown")
axes[1].set_ylabel("Drawdown")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

rolling_12m = strategy_results["net_return"].rolling(12).sum()
rolling_12m.plot(ax=axes[2], color="purple", linewidth=1.8)
axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
axes[2].set_title("Rolling 12-Month Strategy Return")
axes[2].set_ylabel("12M Return")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()