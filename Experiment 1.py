import numpy as np

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

print("Square root:\n", np.sqrt(x))
print("Sum of all elements:", np.sum(y))

print("Column-wise sum:", np.sum(y, axis = 0))
print("Row-wise sum:", np.sum(y, axis = 1))
print("Transpose:\n", x.T)