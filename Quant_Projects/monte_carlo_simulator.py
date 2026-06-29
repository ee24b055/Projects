import numpy as np
import matplotlib.pyplot as plt
initial_portfolio = 500000    # Starting portfolio value
annual_withdrawal = 25000     # Amount withdrawn each year
annual_return_mean = 0.07     # Expected annual return (7%)
annual_return_std = 0.15      # Annual volatility (15%)
n_years = 30                  # Retirement horizon
n_simulations = 10000         # Number of Monte Carlo paths

final_values = []
survival_count = 0
all_paths = []
for _ in range(n_simulations):
    portfolio = initial_portfolio
    path = [portfolio]
    survived = True
    for year in range(n_years):
        annual_return = np.random.normal(annual_return_mean,
                                         annual_return_std)
        portfolio = portfolio * (1 + annual_return) - annual_withdrawal
        path.append(max(portfolio, 0))
        if portfolio <= 0:
            survived = False
            break
    final_values.append(portfolio)
    all_paths.append(path)
    if survived:
        survival_count += 1
survival_rate = survival_count / n_simulations
print(f"Portfolio survival rate: {survival_rate:.1%}")


plt.figure(figsize=(12, 6))
for path in all_paths[:500]:
    color = 'green' if path[-1] > 0 else 'red'
    plt.plot(path, alpha=0.05, color=color, linewidth=0.5)
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
plt.title(f'Monte Carlo Retirement Simulation\n'
          f'Survival Rate: {survival_rate:.1%}')
plt.xlabel('Years')
plt.ylabel('Portfolio Value (£)')
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(final_values, bins=100, edgecolor='black', color='steelblue')
plt.axvline(x=0, color='red', linestyle='--', label='Ruin')
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value (£)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
print(f"5th percentile: £{percentiles[0]:,.0f}")
print(f"25th percentile: £{percentiles[1]:,.0f}")
print(f"Median: £{percentiles[2]:,.0f}")
print(f"75th percentile: £{percentiles[3]:,.0f}")
print(f"95th percentile: £{percentiles[4]:,.0f}")