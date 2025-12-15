import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

normal_data = np.random.normal(loc=0, scale=1, size=1000)

beta_data = np.random.beta(a=2, b=5, size=1000)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(beta_data, bins=20)
plt.title("Beta Distribution (a=2, b=5)")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(normal_data, bins=20)
plt.title("Normal Distribution (μ=0, σ=1)")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()