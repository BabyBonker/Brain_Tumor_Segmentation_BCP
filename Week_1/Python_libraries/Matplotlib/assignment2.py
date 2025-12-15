import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

data = np.random.normal(loc=0, scale=1, size=1000)

mean_val = np.mean(data)
p25 = np.percentile(data, 25)
p75 = np.percentile(data, 75)

plt.figure(figsize=(8, 5))
plt.hist(data, bins=20, color='lightpink', edgecolor='black', alpha=0.8, label='Data')

plt.axvline(mean_val, linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")
plt.axvline(p25, linestyle=':', linewidth=2, label=f"25th Percentile = {p25:.2f}")
plt.axvline(p75, linestyle=':', linewidth=2, label=f"75th Percentile = {p75:.2f}")

plt.title("Histogram of Normally Distributed Data", fontsize=14)
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()
plt.show()