from itertools import product
from fractions import Fraction
def dice_probability(n_dice, n_sides, target_sum):
    total_outcomes = n_sides ** n_dice
    favourable = 0
    for outcome in product(range(1, n_sides + 1), repeat=n_dice):
        if sum(outcome) == target_sum:
            favourable += 1
    return Fraction(favourable, total_outcomes)
print(dice_probability(2, 6, 7))

def biased_dice_probability(n_dice, face_probs, target_sum):
    from itertools import product
    total_prob = 0
    faces = list(range(1, len(face_probs) + 1))
    for outcome in product(faces, repeat=n_dice):
        if sum(outcome) == target_sum:
            prob = 1
            for face in outcome:
                prob *= face_probs[face - 1]
            total_prob += prob
    return total_prob
face_probs = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
print(biased_dice_probability(1, face_probs, 6))

def coupon_collector_simulation(n_sides, n_simulations=100000):
    total_rolls = 0
    for _ in range(n_simulations):
        seen = set()
        rolls = 0
        while len(seen) < n_sides:
            seen.add(np.random.randint(1, n_sides + 1))
            rolls += 1
        total_rolls += rolls
    return total_rolls / n_simulations
print(f"Expected rolls: {coupon_collector_simulation(6):.2f}")


def dice_expected_value(n_dice, n_sides, payoff_function):
    total_ev = 0
    for outcome in product(range(1, n_sides + 1), repeat=n_dice):
        prob = (1 / n_sides) ** n_dice
        total_ev += prob * payoff_function(sum(outcome))
    return total_ev
payoff = lambda x: x ** 2
print(f"Expected value of sum squared: {dice_expected_value(2, 6, payoff):.4f}")