import numpy as np

np.random.seed(0)

poisson_arr = np.random.poisson(lam=5, size=20)

mean_val = poisson_arr.mean()
std_val = poisson_arr.std()

normalized_arr = (poisson_arr - mean_val) / std_val

print("Original Poisson array:")
print(poisson_arr)

print("\nMean:", mean_val)
print("Standard Deviation:", std_val)

print("\nCentered and Normalized array:")
print(normalized_arr)