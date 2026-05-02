import numpy as np
import matplotlib.pyplot as plt

# 1. Sigmoid Activation Function
# Formula: 1 / (1 + exp(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2. ReLU (Rectified Linear Unit)
# Formula: max(0, x)
def relu(x):
    return np.maximum(0, x)

# 3. Tanh (Hyperbolic Tangent)
# Formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
def tanh(x):
    return np.tanh(x)

# 4. Softmax (Commonly used in output layers)
def softmax(x):
    exps = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return exps / np.sum(exps)

# Example: Visualizing Sigmoid (using plotting styles from your lab)
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.show()