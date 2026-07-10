import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.data_fetcher import get_aligned_data
from core.risk_model import calculate_rolling_exposure
from core.optimizer import optimize_factor_neutral_weights

def run_backtest():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'BAC', 'MS', 'GS']
    benchmark = 'SPY'
    
    stock_rets, bench_rets, ff_factors = get_aligned_data(tickers, benchmark, '2021-01-01', '2026-01-01')
    alphas, mkt_b, smb_b, hml_b = calculate_rolling_exposure(stock_rets, ff_factors, window=60)
    
    rebal_dates = alphas.resample('ME').last().index
    rebal_dates = [d for d in rebal_dates if d in alphas.index]
    
    strat_rets = pd.Series(0.0, index=alphas.index)
    current_w = np.zeros(len(tickers))
    bps_cost = 10 / 10000.0 
    
    print("Simulating sequential optimization matrix across time slices...")
    for date in alphas.index:
        if date in rebal_dates:
            new_w = optimize_factor_neutral_weights(
                alphas.loc[date].values, 
                mkt_b.loc[date].values, 
                smb_b.loc[date].values, 
                hml_b.loc[date].values
            )
            turnover = np.sum(np.abs(new_w - current_w))
            strat_rets.loc[date] -= (turnover * bps_cost)
            current_w = new_w
            
        strat_rets.loc[date] += np.dot(current_w, stock_rets.loc[date].values)
        
    bench_aligned = bench_rets.loc[strat_rets.index]
    s_cum = np.exp(strat_rets.cumsum())
    b_cum = np.exp(bench_aligned.cumsum())
    
    s_drawdown = (s_cum - s_cum.cummax()) / s_cum.cummax()
    b_drawdown = (b_cum - b_cum.cummax()) / b_cum.cummax()
    
    metrics = {
        "Annualized Return": [np.exp(strat_rets.mean() * 252) - 1, np.exp(bench_aligned.mean() * 252) - 1],
        "Annualized Volatility": [strat_rets.std() * np.sqrt(252), bench_aligned.std() * np.sqrt(252)],
        "Sharpe Ratio": [(strat_rets.mean() / strat_rets.std()) * np.sqrt(252), (bench_aligned.mean() / bench_aligned.std()) * np.sqrt(252)],
        "Max Drawdown": [s_drawdown.min(), b_drawdown.min()]
    }
    
    print("\n" + "="*65)
    print(" SYSTEMATIC MULTI-FACTOR OPTIMIZATION MATRIX ".center(65, "="))
    print("="*65)
    print(f"{'Performance Metric':<25} | {'Joint-Neutral Strategy':<22} | {'SPY Index':<10}")
    print("-"*65)
    for k, v in metrics.items():
        if "Ratio" in k:
            print(f"{k:<25} | {v[0]:>21.2f} | {v[1]:>10.2f}")
        else:
            print(f"{k:<25} | {v[0]:>21.2%} | {v[1]:>10.2%}")
    print("="*65 + "\n")
    
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(s_cum * 100, label='Joint Factor-Neutral Portfolio (Net of Fees)', color='crimson', lw=2)
    ax.plot(b_cum * 100, label='SPY Index Benchmark', color='slategrey', alpha=0.6, lw=1.5)
    ax.set_title("Institutional Multi-Factor Arbitrage Performance Profile", fontsize=12, fontweight='bold')
    ax.set_ylabel("Compounded Base Asset Growth ($)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()
