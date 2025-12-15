import numpy as np

arr = np.random.rand(3, 6)

add_row = np.array([0, 0, 2, 4, 5, 3])
arr_added = arr + add_row

reshaped = arr_added.reshape(9, 2)

transposed = reshaped.T

mean_val = transposed.mean()

locations = np.where(transposed > mean_val)

print("Transposed Array:\n", transposed)
print("\nMean of the array:", mean_val)
print("\nLocations where elements are greater than the mean:")
print(locations)