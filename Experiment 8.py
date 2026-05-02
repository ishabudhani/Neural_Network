import numpy as np

# 1. Activation functions (built using np.exp found in your lab)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Input data (Example: XOR problem)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 3. Initialize Weights and Biases (using np.random)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

# Weights between input and hidden layer
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
# Weights between hidden and output layer
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

learning_rate = 0.1

# 4. Training Loop
for epoch in range(10000):
    # --- Forward Propagation ---
    # Using np.dot (matrix multiplication) as seen in your lab
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # --- Backpropagation ---
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # --- Updating Weights ---
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

print("Predicted Output after training:")
print(predicted_output)