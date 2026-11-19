## Detailed Analysis Based on the Paper *Adam: A Method for Stochastic Optimization* (arXiv:1412.6980)

---

### 1. Core Algorithm and Detailed Mathematical Notation

#### 1.1 Complete Algorithm Flow

**Adam (Adaptive Moment Estimation) Algorithm**

**Input**: Initial parameters $\mathbf{x}^{(0)}$, hyperparameters $\beta_1, \beta_2 \in (0,1)$, learning rate $\eta > 0$, small constant $\varepsilon > 0$

**Initialization**: $\mathbf{m}^{(0)} = 0$, $\mathbf{v}^{(0)} = 0$, $k = 0$

**Loop until convergence**:

$$
\mathbf{g}^{(k)} = \nabla_{\mathbf{x}} F(\mathbf{x}^{(k)}, \boldsymbol{\xi}^{(k)}) \tag{1}
$$

$$
\mathbf{m}^{(k+1)} = \beta_1 \mathbf{m}^{(k)} + (1 - \beta_1) \mathbf{g}^{(k)} \tag{2}
$$

$$
\mathbf{v}^{(k+1)} = \beta_2 \mathbf{v}^{(k)} + (1 - \beta_2) (\mathbf{g}^{(k)})^2 \tag{3}
$$

$$
\hat{\mathbf{m}}^{(k+1)} = \frac{\mathbf{m}^{(k+1)}}{1 - \beta_1^{k+1}}, \quad \hat{\mathbf{v}}^{(k+1)} = \frac{\mathbf{v}^{(k+1)}}{1 - \beta_2^{k+1}} \tag{4}
$$

$$
\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} - \eta \frac{\hat{\mathbf{m}}^{(k+1)}}{\sqrt{\hat{\mathbf{v}}^{(k+1)}} + \varepsilon} \tag{5}
$$

---

#### 1.2 Detailed Explanation of Each Mathematical Symbol

| Symbol | Meaning | Concrete Example |
|------|------|----------|
| $\mathbf{x}^{(k)}$ | **Model parameter vector** at iteration $k$ | When training a neural network, $\mathbf{x}$ represents all neuron weights and biases. For example: $\mathbf{x} = [w_1, w_2, ..., w_n]^T$ |
| $\mathbf{g}^{(k)}$ | **Gradient vector** at iteration $k$ | If the loss function is $F(\mathbf{x}) = (x_1 - 3)^2 + (x_2 + 1)^2$, then $\mathbf{g} = [2(x_1-3), 2(x_2+1)]^T$ |
| $\boldsymbol{\xi}^{(k)}$ | **Stochastic mini-batch sample** at iteration $k$ | A randomly drawn subset of 32 or 64 samples from the training dataset |
| $\mathbf{m}^{(k)}$ | **First moment estimate** at iteration $k$ (exponentially weighted moving average of gradients) | This is the "smoothed" version of the gradient. If past gradients are $[2, -1, 3]$, then $\mathbf{m}$ is their weighted average |
| $\mathbf{v}^{(k)}$ | **Second moment estimate** at iteration $k$ (exponentially weighted moving average of squared gradients) | If the gradient is $g=2$, then $g^2=4$, and $\mathbf{v}$ is the weighted average of these squared values |
| $(\mathbf{g}^{(k)})^2$ | **Element-wise square** (Hadamard square) | If $\mathbf{g} = [2, -3, 1]$, then $(\mathbf{g})^2 = [4, 9, 1]$ |
| $\beta_1, \beta_2$ | **Exponential decay rates**, controlling the forgetting of historical information | Default values are $\beta_1 = 0.9$, $\beta_2 = 0.999$ |
| $\hat{\mathbf{m}}^{(k)}$ | **Bias-corrected first moment estimate** | Since the initial $\mathbf{m}^{(0)}=0$ causes underestimation, we divide by $1-\beta_1^k$ for correction |
| $\hat{\mathbf{v}}^{(k)}$ | **Bias-corrected second moment estimate** | Similarly, we divide by $1-\beta_2^k$ for correction |
| $\eta$ | **Learning rate** (step size), controlling the magnitude of parameter updates | Default is $\eta = 10^{-3}$ |
| $\varepsilon$ | **Small constant** to prevent division by zero | Default is $\varepsilon = 10^{-8}$ |

---

### 2. Connections of First and Second Moments to Physics and Statistics

#### 2.1 First Moment: Momentum (Physics) and Mean (Statistics)

**Statistical connection**:

In statistics, the **$r$-th moment** of a random variable $X$ is defined as $E[X^r]$. The **first moment** $E[X]$ is exactly the **mean** (expectation).

In Adam, $\mathbf{m}^{(k+1)} = \beta_1 \mathbf{m}^{(k)} + (1-\beta_1)\mathbf{g}^{(k)}$ is the **exponentially weighted moving average** of the gradient. Essentially, it is a **weighted mean estimate** of past gradients.

**Physical connection — Momentum**:

In Newtonian mechanics, **momentum** is $\mathbf{p} = m\mathbf{v}$ (mass × velocity). The update rule for SGD with Momentum is:

$$
\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla f(\mathbf{x}_t), \quad \mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{v}_{t+1}
$$

where $\mathbf{v}$ is the "velocity" and $\gamma \mathbf{v}_t$ is the "inertia term".

Adam's $\mathbf{m}^{(k+1)} = \beta_1 \mathbf{m}^{(k)} + (1-\beta_1)\mathbf{g}^{(k)}$ is formally identical:
- $\mathbf{m}^{(k)}$ acts as **velocity/momentum**
- $\beta_1$ acts as the **friction coefficient** (smaller $1-\beta_1$ means more momentum is retained)
- This embodies the physical intuition of a **"heavy ball rolling down a hill"**

**First moment = Momentum = Mean**: All three are fundamentally "**weighted historical averages**".

#### 2.2 Second Moment: Kinetic Energy (Physics) and Variance/Scale (Statistics)

**Statistical connection**:

The **second moment** $E[X^2]$ is closely related to the **variance** $\text{Var}(X) = E[X^2] - (E[X])^2$.

In Adam, $\mathbf{v}^{(k+1)} = \beta_2 \mathbf{v}^{(k)} + (1-\beta_2)(\mathbf{g}^{(k)})^2$ is the weighted average of squared gradients. $\sqrt{\hat{\mathbf{v}}}$ is approximately the **standard deviation** of the gradient, used to **normalize** the parameter update.

**Physical connection — Kinetic Energy**:

**Kinetic energy** is $E_k = \frac{1}{2}mv^2$, which is proportional to the **square** of velocity. Adam's $(\mathbf{g}^{(k)})^2$ is analogous—it measures the "**intensity**" or "**energy**" of the gradient variation.

More importantly, $\sqrt{\hat{\mathbf{v}}}$ serves a **normalization** role in the denominator:
- **Large** gradient → **large** $\hat{\mathbf{v}}$ → update step size **decreases** ("braking")
- **Small** gradient → **small** $\hat{\mathbf{v}}$ → update step size **increases** ("accelerating")

This resembles adaptive control in physics where **high kinetic energy triggers deceleration, and low kinetic energy triggers acceleration**.

**Second moment = Kinetic energy indicator = Variance/Scale**: All three measure the "**intensity of variation**" and are used to adaptively adjust the update step size.

| Concept | Role in Adam | Physical Analogy | Statistical Analogy |
|------|--------------|---------|---------|
| **First moment** $\mathbf{m}$ | Weighted average of gradients (direction) | Momentum $\mathbf{p}=m\mathbf{v}$ | Mean $E[X]$ |
| **Second moment** $\mathbf{v}$ | Weighted average of squared gradients (magnitude) | Kinetic energy $E_k \propto v^2$ | Second moment $E[X^2]$ |
| $\sqrt{\hat{\mathbf{v}}}$ | Adaptive normalization denominator | Adjustment of inertia/mass | Standard deviation $\sigma$ |

---

### 3. Visualization Code

```python
import numpy as np
import matplotlib.pyplot as plt

# ============ Define the objective function ============
def f(x, y):
    """Complex terrain: a function with local minima"""
    return 0.5 * (x**2 + y**2) + 0.3 * np.sin(3*x) * np.cos(3*y) + 0.2 * (x + y)

def grad_f(x, y):
    """Gradient (analytical solution)"""
    dx = x + 0.9 * np.cos(3*x) * np.cos(3*y) + 0.2
    dy = y - 0.9 * np.sin(3*x) * np.sin(3*y) + 0.2
    return np.array([dx, dy])

# ============ Adam Optimizer ============
class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None      # First moment
        self.v = None      # Second moment
        self.t = 0         # Time step
    
    def step(self, theta, g):
        """Perform one Adam update step"""
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(g)
            self.v = np.zeros_like(g)
        
        # Update first moment (momentum)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        # Update second moment (adaptive scale)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Parameter update
        theta_new = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return theta_new

# ============ Run optimization ============
np.random.seed(42)
theta = np.array([1.2, -0.8])  # Initial position
optimizer = Adam(lr=0.15)

# Store trajectory
trajectory = [theta.copy()]
m_history = []      # First moment history
v_history = []      # Second moment history

for i in range(50):
    g = grad_f(theta[0], theta[1])
    theta = optimizer.step(theta, g)
    trajectory.append(theta.copy())
    m_history.append(optimizer.m.copy())
    v_history.append(optimizer.v.copy())

trajectory = np.array(trajectory)
m_history = np.array(m_history)
v_history = np.array(v_history)

# ============ Plotting ============
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Optimization path
x_vals = np.linspace(-2, 2, 200)
y_vals = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

ax1 = axes[0, 0]
contour = ax1.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
ax1.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2, markersize=8, label='Adam path')
ax1.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start point')
ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'rs', markersize=10, label='End point')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_title('(a) Adam Optimization Path')
ax1.legend(); ax1.grid(alpha=0.3)

# 2. First moment (momentum) evolution
ax2 = axes[0, 1]
steps = np.arange(1, len(m_history) + 1)
ax2.plot(steps, m_history[:, 0], 'b-', label='$m_x$ (x-direction momentum)')
ax2.plot(steps, m_history[:, 1], 'orange', label='$m_y$ (y-direction momentum)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Iteration step'); ax2.set_ylabel('First moment $m$')
ax2.set_title('(b) First Moment (Momentum) Evolution')
ax2.legend(); ax2.grid(alpha=0.3)

# 3. Second moment evolution
ax3 = axes[0, 2]
ax3.plot(steps, v_history[:, 0], 'b-', label='$v_x$ (x-direction scale)')
ax3.plot(steps, v_history[:, 1], 'orange', label='$v_y$ (y-direction scale)')
ax3.set_xlabel('Iteration step'); ax3.set_ylabel('Second moment $v$')
ax3.set_title('(c) Second Moment (Adaptive Scale) Evolution')
ax3.legend(); ax3.grid(alpha=0.3)

# 4. Gradient vs momentum comparison
ax4 = axes[1, 0]
gradients = []
for i in range(50):
    g = grad_f(trajectory[i, 0], trajectory[i, 1])
    gradients.append(g)
gradients = np.array(gradients)
ax4.plot(steps, gradients[:, 0], 'b--', alpha=0.6, label='Raw gradient $g_x$')
ax4.plot(steps, m_history[:, 0], 'b-', linewidth=2, label='First moment $m_x$ (smoothed)')
ax4.set_xlabel('Iteration step'); ax4.set_ylabel('Gradient / First moment')
ax4.set_title('(d) Gradient vs First Moment (Smoothing Effect)')
ax4.legend(); ax4.grid(alpha=0.3)

# 5. Gradient squared vs second moment comparison
ax5 = axes[1, 1]
g_sq_x = gradients[:, 0] ** 2
ax5.plot(steps, g_sq_x, 'b--', alpha=0.6, label='$g_x^2$ (raw)')
ax5.plot(steps, v_history[:, 0], 'b-', linewidth=2, label='$v_x$ (smoothed)')
ax5.set_xlabel('Iteration step'); ax5.set_ylabel('Gradient squared / Second moment')
ax5.set_title('(e) Gradient Squared vs Second Moment (Smoothing Effect)')
ax5.legend(); ax5.grid(alpha=0.3)

# 6. Adaptive learning rate
ax6 = axes[1, 2]
v_hat_history = v_history / (1 - optimizer.beta2 ** steps[:, None])
adaptive_lr_x = optimizer.lr / (np.sqrt(v_hat_history[:, 0]) + optimizer.eps)
adaptive_lr_y = optimizer.lr / (np.sqrt(v_hat_history[:, 1]) + optimizer.eps)
ax6.plot(steps, adaptive_lr_x, 'b-', label='Effective LR (x-direction)')
ax6.plot(steps, adaptive_lr_y, 'orange', label='Effective LR (y-direction)')
ax6.axhline(y=optimizer.lr, color='gray', linestyle='--', alpha=0.5, label='Original LR')
ax6.set_xlabel('Iteration step'); ax6.set_ylabel('Effective learning rate')
ax6.set_title(r'(f) Adaptive Learning Rate ($\eta / (\sqrt{\hat v} + \varepsilon)$)')
ax6.legend(); ax6.grid(alpha=0.3)

plt.suptitle('Adam Optimization Visualization: First Moment (Momentum) and Second Moment (Adaptive Scale)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Code Output Interpretation

Running the above code generates a 2×3 grid of subplots:

| Subplot | Content | Key Observation |
|------|---------|------|
| (a) Optimization path | Adam's trajectory in complex terrain | The red path smoothly reaches the endpoint from the start |
| (b) First moment | Momentum $m_x, m_y$ over iterations | Momentum accumulates historical gradient direction, acting as "inertia" |
| (c) Second moment | Adaptive scale $v_x, v_y$ over iterations | When gradients are large, $v$ increases, automatically applying "brakes" |
| (d) Gradient vs first moment | Raw gradient vs smoothed momentum | The first moment filters out gradient noise, providing more stability |
| (e) Gradient squared vs second moment | Raw gradient squared vs smoothed second moment | The second moment provides a stable estimate of gradient magnitude trends |
| (f) Adaptive learning rate | $\eta/(\sqrt{\hat v}+\varepsilon)$ over iterations | Different directions have different effective learning rates |

---

### 4. Academic Sources

- **Original Paper**: Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *3rd International Conference for Learning Representations*, San Diego
- **arXiv Link**: https://arxiv.org/abs/1412.6980
- **Cornell University Optimization Materials**: Adam is an extension of SGD, combining Momentum and RMSProp
- **PennyLane Documentation**: Explicitly analogizes the first moment to "momentum" and the second moment to "velocity"
- **Physical Analogy**: The momentum term in Adam originates from the physical intuition of a "heavy ball rolling down a hill"
