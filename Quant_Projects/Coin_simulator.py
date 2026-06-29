# #Let work on coin flip simulator, to understand LLN and CLT
import numpy as np
# import matplotlib.pyplot as plt

# n_flips = 10000
# n_simulations = 100000 #number of possible alternative universes (scenarios) you are testing.

# results = []
# for i in range(n_simulations):
#     flips = np.random.choice(['H', 'T'], size=n_flips)
#     heads_count = np.sum(flips == 'H')
#     tails_count = np.sum(flips == 'T')
#     results.append((heads_count, tails_count))

# # plt.hist(results, bins=50, edgecolor='black', color='steelblue')
# # plt.axvline(x=5000, color='red', linestyle='--', label='Expected value')
# # plt.title('Distribution of Heads in 10,000 Coin Flips')
# # plt.xlabel('Number of Heads')
# # plt.ylabel('Frequency')
# # plt.legend()
# # plt.show()
# # plt.hist([heads for heads, tails in results], bins=50, edgecolor='black', color='steelblue')
# # plt.axvline(x=n_flips/2, color='red', linestyle='--', label='Expected value')
# # plt.title(f'Distribution of Heads in {n_flips} Coin Flips ({n_simulations} Simulations)')
# # plt.xlabel('Number of Heads')
# # plt.ylabel('Frequency')
# # plt.legend()
# # plt.show()

# probability_empirical = np.mean(np.array(results) > 5200)
# print(f"Empirical probability: {probability_empirical:.4f}")

# from scipy import stats
# probability_analytical = 1 - stats.norm.cdf(4.0)
# print(f"Analytical probability: {probability_analytical:.6f}")

def monty_hall_simulation(switch, n_simulations=100000):
    wins = 0
    for _ in range(n_simulations):
        car = np.random.randint(0, 3)
        choice = np.random.randint(0, 3)
        if switch:
            wins += (choice != car)
        else:
            wins += (choice == car)
    return wins / n_simulations
print(f"Win rate (stay): {monty_hall_simulation(switch=False):.3f}")
print(f"Win rate (switch): {monty_hall_simulation(switch=True):.3f}")