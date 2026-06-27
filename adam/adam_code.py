import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============ Define the objective function ============
def f(x, y):
    """Complex terrain: a function with local minima"""
    return 0.5 * (x**2 + y**2) + 0.3 * np.sin(3*x) * np.cos(3*y) + 0.2 * (x + y)

def grad_f(x, y):
    """Gradient (analytical solution)"""
    dx = x - 0.9 * np.sin(3*x) * np.cos(3*y) + 0.2
    dy = y + 0.9 * np.cos(3*x) * np.sin(3*y) + 0.2
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
adaptive_lr_x = optimizer.lr / (np.sqrt(v_history[:, 0]) + optimizer.eps)
adaptive_lr_y = optimizer.lr / (np.sqrt(v_history[:, 1]) + optimizer.eps)
ax6.plot(steps, adaptive_lr_x, 'b-', label='Effective LR (x-direction)')
ax6.plot(steps, adaptive_lr_y, 'orange', label='Effective LR (y-direction)')
ax6.axhline(y=optimizer.lr, color='gray', linestyle='--', alpha=0.5, label='Original LR')
ax6.set_xlabel('Iteration step'); ax6.set_ylabel('Effective learning rate')
ax6.set_title('(f) Adaptive Learning Rate ($\\eta / \\sqrt{v}$)')
ax6.legend(); ax6.grid(alpha=0.3)

plt.suptitle('Adam Optimization Visualization: First Moment (Momentum) and Second Moment (Adaptive Scale)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
