import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import comb

# ============================================================
# 1. Path length distribution (Binomial distribution)
# ============================================================
def plot_path_length_distribution(n_layers=110, ax=None):
    """Plot the binomial distribution of ResNet path lengths"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    k = np.arange(0, n_layers + 1)
    # Binomial distribution: P(L=k) = C(n,k) * 0.5^n
    probs = binom.pmf(k, n_layers, 0.5)
    
    ax.bar(k, probs, width=0.8, color='steelblue', alpha=0.7, 
           edgecolor='navy', linewidth=0.5)
    ax.axvline(n_layers / 2, color='red', linestyle='--', linewidth=2,
               label=f'Mean = {n_layers/2:.1f}')
    ax.axvline(34, color='orange', linestyle='-.', linewidth=2,
               label='Upper bound of effective paths (34 layers)')
    ax.axvline(10, color='orange', linestyle='-.', linewidth=2,
               label='Lower bound of effective paths (10 layers)')
    
    ax.set_xlabel('Path length (number of residual blocks traversed)', fontsize=12)
    ax.set_ylabel('Probability density', fontsize=12)
    ax.set_title(f'ResNet-{n_layers} Path Length Distribution (Binomial)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, n_layers)
    
    return ax

# ============================================================
# 2. Gradient contribution distribution (simulating the paper's findings)
# ============================================================
def plot_gradient_contribution(n_layers=110, ax=None):
    """
    Simulate the gradient contribution distribution.
    The paper finds that most gradients come from paths of length 10–34.
    This uses a truncated normal distribution to simulate that phenomenon.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    k = np.arange(0, n_layers + 1)
    
    # Simulate gradient contribution: concentrated around layers 10–34
    # Using a truncated normal distribution, mean=22, std=8
    from scipy.stats import truncnorm
    
    a = (0 - 22) / 8
    b = (n_layers - 22) / 8
    grad_contrib = truncnorm.pdf(k, a, b, loc=22, scale=8)
    grad_contrib = grad_contrib / grad_contrib.sum()  # Normalize
    
    ax.fill_between(k, grad_contrib, alpha=0.5, color='coral', 
                    label='Gradient contribution distribution (simulated)')
    ax.plot(k, grad_contrib, 'r-', linewidth=2)
    
    # Mark the 10–34 layer region
    ax.axvspan(10, 34, alpha=0.15, color='green', 
               label='Primary gradient contribution region (10–34 layers)')
    ax.axvline(10, color='green', linestyle='--', alpha=0.7)
    ax.axvline(34, color='green', linestyle='--', alpha=0.7)
    
    # Annotate: "most gradients come from short paths"
    ax.annotate('~95% gradients come from\n10–34 layer paths',
                xy=(22, grad_contrib[22] * 0.6),
                xytext=(50, grad_contrib[50] * 0.8),
                arrowprops=dict(arrowstyle='->', color='darkred'),
                fontsize=11, color='darkred')
    
    ax.set_xlabel('Path length (layers)', fontsize=12)
    ax.set_ylabel('Gradient contribution fraction', fontsize=12)
    ax.set_title(f'ResNet-{n_layers}: Gradient Contribution vs. Path Length', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, n_layers)
    ax.set_ylim(0, grad_contrib.max() * 1.2)
    
    return ax

# ============================================================
# 3. Exponential growth of the number of paths
# ============================================================
def plot_path_count_growth(ax=None):
    """Show how the number of paths grows exponentially with depth"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    n = np.arange(1, 21)
    num_paths = 2 ** n
    
    ax.semilogy(n, num_paths, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of residual blocks (n)', fontsize=12)
    ax.set_ylabel('Number of paths (2ⁿ)', fontsize=12)
    ax.set_title('ResNet Path Count Grows Exponentially with Depth', fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xticks(n)
    
    # Annotate key points
    for i in [5, 10, 15, 20]:
        ax.annotate(f'2^{i} = {2**i:,}',
                    xy=(i, 2**i),
                    xytext=(i-1, 2**i * 1.5),
                    fontsize=9)
    
    return ax

# ============================================================
# 4. Combined visualization (main figure)
# ============================================================
def create_full_visualization(n_layers=110):
    """Create a complete 3-in-1 visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Path length distribution
    plot_path_length_distribution(n_layers, axes[0])
    
    # Subplot 2: Gradient contribution distribution
    plot_gradient_contribution(n_layers, axes[1])
    
    # Subplot 3: Path count growth
    plot_path_count_growth(axes[2])
    
    plt.suptitle('Core Evidence that ResNet Acts as an Ensemble of Shallow Networks\n'
                 '(based on Veit et al., NIPS 2016 "Residual Networks Behave Like Ensembles of Relatively Shallow Networks")',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. Bonus: Path enumeration visualization (small scale)
# ============================================================
def visualize_paths_small(n=4):
    """
    Visualize all paths in a small-scale ResNet (n=4)
    Demonstrates the concept of the "Unraveled View"
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Draw all 2^n paths
    num_paths = 2 ** n
    y_positions = np.arange(num_paths)
    
    for path_idx in range(num_paths):
        # Convert path_idx to binary: 1 = pass through this block, 0 = skip
        binary = [int(b) for b in format(path_idx, f'0{n}b')]
        path_length = sum(binary)
        
        # Calculate the indices of active layers
        active_layers = [i for i, active in enumerate(binary) if active]
        
        # Draw the path: from x=0 to x=n, fixed y
        x_coords = []
        y_coords = []
        current_x = 0
        
        for layer_idx, active in enumerate(binary):
            if active:
                # Pass through this layer: shift upward
                x_coords.append(layer_idx)
                y_coords.append(y_positions[path_idx])
                x_coords.append(layer_idx + 0.5)
                y_coords.append(y_positions[path_idx] + 0.3 * (path_length / n))
                x_coords.append(layer_idx + 0.5)
                y_coords.append(y_positions[path_idx] + 0.3 * (path_length / n))
                x_coords.append(layer_idx + 1)
                y_coords.append(y_positions[path_idx])
            else:
                # Skip this layer: straight horizontal connection
                x_coords.append(layer_idx)
                y_coords.append(y_positions[path_idx])
                x_coords.append(layer_idx + 1)
                y_coords.append(y_positions[path_idx])
        
        # Color varies with path length
        color = plt.cm.viridis(path_length / n)
        ax.plot(x_coords, y_coords, '-', color=color, linewidth=1.5, alpha=0.7)
        
        # Annotate path length on the right
        ax.text(n + 0.2, y_positions[path_idx], 
                f'len={path_length}', fontsize=8, va='center')
    
    # Draw residual block boundaries
    for i in range(n + 1):
        ax.axvline(i, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        if i < n:
            ax.text(i + 0.5, -1.5, f'Block {i+1}', ha='center', fontsize=10)
    
    ax.set_xlim(-0.5, n + 1.5)
    ax.set_ylim(-2, num_paths)
    ax.set_xlabel('Residual block index', fontsize=12)
    ax.set_ylabel('Path index', fontsize=12)
    ax.set_title(f'Unraveled View: An n={n} ResNet Contains {num_paths} Paths\n'
                 '(Brighter color indicates longer path)', fontsize=14)
    ax.set_yticks([])
    ax.grid(alpha=0.1)
    
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                                norm=plt.Normalize(vmin=0, vmax=n))
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', 
                        pad=0.02, aspect=30)
    cbar.set_label('Path length (number of residual blocks traversed)', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# ============================================================
# Execute visualizations
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ResNet Ensemble Perspective Visualization")
    print("Based on: Veit et al., NIPS 2016")
    print("=" * 60)
    
    # Main figure: 3-in-1
    create_full_visualization(n_layers=110)
    
    # Small-scale path enumeration
    visualize_paths_small(n=4)
    
    print("\nKey Insights:")
    print("1. Path lengths follow a binomial distribution with mean = n/2")
    print("2. Most gradients come from short paths (10–34 layers)")
    print("3. The number of paths grows exponentially with depth (2ⁿ)")
    print("4. ResNet is essentially an ensemble of shallow networks")
