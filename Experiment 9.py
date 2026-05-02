import numpy as np

def adam_optimizer(w, dw, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    # Step 1: Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * dw
    
    # Step 2: Update biased second raw moment estimate (using power from your lab)
    v = beta2 * v + (1 - beta2) * np.power(dw, 2)
    
    # Step 3: Compute bias-corrected estimates
    m_hat = m / (1 - np.power(beta1, t))
    v_hat = v / (1 - np.power(beta2, t))
    
    # Step 4: Update weights (using sqrt from your lab)
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return w, m, v

# --- Test Case: Minimizing a Quadratic Function f(x) = x^2 ---
# The goal is for the weight (x) to reach 0.0

# Initializing parameters
weight = 10.0 # Starting point
m = 0.0
v = 0.0
iterations = 100

print(f"{'Iteration':<15} | {'Weight (x)':<15} | {'Gradient (dw)':<15}")
print("-" * 50)

for t in range(1, iterations + 1):
    # Derivative of x^2 is 2x
    gradient = 2 * weight
    
    # Apply Adam Update
    weight, m, v = adam_optimizer(weight, gradient, m, v, t)
    
    # Print output every 20 iterations to show progress
    if t % 20 == 0 or t == 1:
        print(f"{t:<15} | {weight:<15.6f} | {gradient:<15.6f}")

print("-" * 50)
print(f"Final Weight after {iterations} iterations: {weight:.10f}")