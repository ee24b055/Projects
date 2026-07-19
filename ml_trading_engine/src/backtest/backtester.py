import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(test_df: pd.DataFrame):
    df = test_df.copy()
    
    # Strategy execution: Shift prediction by 1 day because today's signal applies to tomorrow's market execution
    df['Strategy_Return'] = df['Predicted_Signal'].shift(1) * df['Return']
    df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
    
    # Compounding performance growth
    df['Cum_Benchmark'] = (1 + df['Return']).cumprod()
    df['Cum_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    
    total_strat_return = df['Cum_Strategy'].iloc[-1] - 1 # need last row's value
    total_bench_return = df['Cum_Benchmark'].iloc[-1] - 1
    #diff of above two gives alpha
    # Quantitative Risk Assessment (Sharpe Ratio calculation)
    sharpe = (df['Strategy_Return'].mean() / (df['Strategy_Return'].std() + 1e-9)) * np.sqrt(252)
    
    print("\n" + "="*30 + "\n   BACKTEST METRICS\n" + "="*30)
    print(f"ML Strategy Return : {total_strat_return:.2%}")
    print(f"Benchmark Return   : {total_bench_return:.2%}")
    print(f"Strategy Sharpe    : {sharpe:.2f}")
    print("="*30)  # to make output look good
    
    # Save visualization plot locally
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Cum_Benchmark'], label='Benchmark (Buy & Hold)', color='orange')
    plt.plot(df.index, df['Cum_Strategy'], label='ML Automated Strategy', color='blue')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_chart.png')
    print("[+] Saved 'performance_chart.png' in root directory.")
