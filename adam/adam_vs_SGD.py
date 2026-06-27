import numpy as np
import matplotlib.pyplot as plt

# Define a 2D elongated quadratic valley (f(x,y) = x^2 + 10y^2)
def loss_function(x, y):
    return x**2 + 10 * y**2

# True gradients for the function
def gradients(x, y):
    return np.array([2*x, 20*y])

# Simulation parameters
start_point = np.array([-8.0, 3.0])
iterations = 50
alpha = 0.5 # Learning rate

# --- Standard SGD ---
sgd_path = [start_point]
theta = start_point.copy()
for _ in range(iterations):
    grad = gradients(theta[0], theta[1])
    theta = theta - (alpha / 5) * grad # Scaled down alpha so it doesn't explode
    sgd_path.append(theta.copy())

# --- Adam Optimizer ---
adam_path = [start_point]
theta = start_point.copy()
m = np.zeros(2)
v = np.zeros(2)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

for t in range(1, iterations + 1):
    grad = gradients(theta[0], theta[1])
    
    m = beta1 * m + (1 - beta1) * grad          # 1st moment
    v = beta2 * v + (1 - beta2) * (grad**2)     # 2nd moment
    
    m_hat = m / (1 - beta1**t)                  # Bias correction
    v_hat = v / (1 - beta2**t)
    
    theta = theta - alpha * (m_hat / (np.sqrt(v_hat) + epsilon))
    adam_path.append(theta.copy())

# --- Plotting ---
sgd_path = np.array(sgd_path)
adam_path = np.array(adam_path)

x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-5, 5, 100))
z = loss_function(x, y)

plt.figure(figsize=(10, 6))
plt.contour(x, y, z, levels=np.logspace(-1, 3, 20), cmap='gray_r', alpha=0.5)
plt.plot(sgd_path[:, 0], sgd_path[:, 1], 'ro-', label='SGD (Oscillates & Slow)', markersize=4)
plt.plot(adam_path[:, 0], adam_path[:, 1], 'bo-', label='Adam (Damped & Fast)', markersize=4)
plt.plot(0, 0, 'y*', markersize=15, label='Global Minimum')

plt.title("Adam vs SGD on an Elongated Valley")
plt.xlabel("x (Flat direction)")
plt.ylabel("y (Steep direction)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

