# Weighted Objective Function (Augmented Chebyshev)
def calculate_weighted_objective(weights, values, rho):
    weighted_vals = [value * weight for value, weight in zip(values,weights)] 
    aug_tcheby = max(weighted_vals) + rho * sum(weighted_vals)
    return aug_tcheby