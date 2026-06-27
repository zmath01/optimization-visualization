## Paper Analysis: "Attention Is All You Need"

**Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

**Affiliation**: Google Brain & Google Research

**Published in**: *31st Conference on Neural Information Processing Systems (NIPS 2017)*, Long Beach, CA, USA

**arXiv**: 1706.03762

**Core Thesis**: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. We propose a new simple network architecture, the **Transformer**, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

---

## 1. Core Ideas and Key Formulas

### 1.1 The Transformer Architecture

The Transformer follows the encoder-decoder structure using **stacked self-attention** and **point-wise fully connected layers** for both the encoder and decoder. The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers:
1. A **multi-head self-attention** mechanism
2. A simple, **position-wise fully connected feed-forward network**

Each sub-layer employs a **residual connection** followed by **layer normalization**:

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
\tag{1}
$$

### 1.2 Scaled Dot-Product Attention

The core attention mechanism is **Scaled Dot-Product Attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\tag{2}
$$

where:
- $Q$ (queries), $K$ (keys), and $V$ (values) are matrices
- $d_k$ is the dimension of the keys

### 1.3 Multi-Head Attention

Instead of performing a single attention function, the model linearly projects the queries, keys, and values $h$ times with different learned projections:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\tag{3}
$$

where:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\tag{4}
$$

The projections are parameter matrices:
- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$

### 1.4 Positional Encoding

Since the model contains no recurrence and no convolution, it must inject information about the relative or absolute position of tokens in the sequence. The paper uses **sinusoidal positional encodings**:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\tag{5}
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\tag{6}
$$

where:
- $pos$ is the position in the sequence
- $i$ is the dimension index
- $d_{\text{model}}$ is the model dimension

### 1.5 Position-wise Feed-Forward Networks

Each layer contains a fully connected feed-forward network applied to each position separately and identically:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
\tag{7}
$$

with ReLU activation.

### 1.6 Model Hyperparameters

The Transformer base model uses:

| Parameter | Value | Description |
|-----------|-------|-------------|
| $N$ | 6 | Number of encoder/decoder layers |
| $d_{\text{model}}$ | 512 | Model dimension |
| $h$ | 8 | Number of attention heads |
| $d_k = d_v$ | 64 | Key/value dimension per head |
| $d_{ff}$ | 2048 | Inner dimension of feed-forward network |
| $P_{\text{drop}}$ | 0.1 | Dropout rate |
| $\epsilon_{ls}$ | 0.1 | Label smoothing value |

---

## 2. Detailed Explanation of Mathematical Symbols

### Table 1: Scaled Dot-Product Attention Symbols

| Symbol | Meaning | Concrete Example |
|------|------|----------|
| $Q$ | **Query matrix** containing queries for each position | For translating "I love you" (3 tokens, $d_k=64$), $Q \in \mathbb{R}^{3 \times 64}$. Each row is a query vector representing "what am I looking for?" |
| $K$ | **Key matrix** containing keys for each position | For the same sentence, $K \in \mathbb{R}^{3 \times 64}$. Each row is a key vector representing "what do I have to offer?" |
| $V$ | **Value matrix** containing values for each position | $V \in \mathbb{R}^{3 \times d_v}$. Each row contains the actual information to be aggregated |
| $QK^T$ | **Dot-product** between queries and keys | Computes attention scores. If $Q$ is row $i$ and $K$ is row $j$, $Q_i \cdot K_j$ measures how much position $i$ should attend to position $j$ |
| $\sqrt{d_k}$ | **Scaling factor** | $d_k = 64$, so $\sqrt{64} = 8$. Prevents dot products from growing too large, which would push softmax into regions with extremely small gradients |
| $\text{softmax}(\cdot)$ | Normalizes attention scores to probabilities | Converts scores to a probability distribution over keys for each query |
| $V$ | Multiplied by attention weights | Produces weighted sum of values: $\sum_j \text{weight}_{ij} \cdot V_j$ |

**Example**: Suppose translating "I love you" to French. For the query "love" attending to "you":
- $Q_{\text{love}} \cdot K_{\text{you}} = 12.8$ (high score → strong attention)
- $Q_{\text{love}} \cdot K_{\text{I}} = -2.1$ (low score → weak attention)
- After softmax: weights ≈ [0.02, 0.05, 0.93] (93% attention to "you")
- Output = weighted sum of value vectors

### Table 2: Multi-Head Attention Symbols

| Symbol | Meaning | Concrete Example |
|------|------|----------|
| $h$ | **Number of attention heads** | $h = 8$ in the base model. Each head learns to attend to different types of relationships |
| $\text{head}_i$ | **Attention output** from the $i$-th head | head$_1$ might attend to syntactic relationships, head$_2$ to semantic relationships, etc. |
| $W_i^Q$ | **Query projection matrix** for head $i$ | $W_i^Q \in \mathbb{R}^{512 \times 64}$. Projects the 512-dimensional input to a 64-dimensional query space |
| $W_i^K$ | **Key projection matrix** for head $i$ | Same dimensions, projects to key space |
| $W_i^V$ | **Value projection matrix** for head $i$ | $W_i^V \in \mathbb{R}^{512 \times d_v}$, projects to value space |
| $W^O$ | **Output projection matrix** | $W^O \in \mathbb{R}^{h d_v \times 512}$. Combines outputs from all heads back to 512 dimensions |
| $\text{Concat}$ | **Concatenation** operation | Stacks the $h$ head outputs along the dimension axis: if each head outputs $64$ dimensions, concatenation gives $8 \times 64 = 512$ dimensions |

**Example**: With $h=8$ heads, each head computes attention in a different 64-dimensional subspace:
- Head 1: "love" attends strongly to "you" (object relation)
- Head 2: "love" attends moderately to "I" (subject relation)
- Head 3: "love" attends weakly to itself (self-attention)
- ...
- All 8 heads' outputs are concatenated and projected to produce the final output

### Table 3: Positional Encoding Symbols

| Symbol | Meaning | Concrete Example |
|------|------|----------|
| $pos$ | **Position** in the sequence | For token 0 ("I"), $pos=0$; token 1 ("love"), $pos=1$; token 2 ("you"), $pos=2$ |
| $i$ | **Dimension index** | For $d_{\text{model}}=512$, $i$ ranges from 0 to 255 (since each position has sine for even and cosine for odd dimensions) |
| $PE_{(pos, 2i)}$ | **Sine encoding** for even dimensions | For $pos=1$, $i=0$: $PE_{(1,0)} = \sin(1 / 10000^{0/512}) = \sin(1) \approx 0.84$ |
| $PE_{(pos, 2i+1)}$ | **Cosine encoding** for odd dimensions | For $pos=1$, $i=0$: $PE_{(1,1)} = \cos(1 / 10000^{0/512}) = \cos(1) \approx 0.54$ |
| $10000^{2i/d_{\text{model}}}$ | **Frequency scaling** | Creates different wavelengths. For $i=0$: $10000^0 = 1$ (fast changing); for $i=255$: $10000^{510/512} \approx 10000$ (slow changing) |

**Example**: For position $pos=3$, dimension $i=2$:
- $PE_{(3,4)} = \sin(3 / 10000^{4/512}) = \sin(3 / 10000^{0.0078125}) = \sin(3 / 1.072) \approx \sin(2.80) \approx 0.34$

### Table 4: Feed-Forward Network Symbols

| Symbol | Meaning | Concrete Example |
|------|------|----------|
| $x$ | **Input** to the FFN | For a token embedding of dimension 512, $x \in \mathbb{R}^{512}$ |
| $W_1$ | **First weight matrix** | $W_1 \in \mathbb{R}^{512 \times 2048}$ (expands dimension) |
| $b_1$ | **First bias** | $b_1 \in \mathbb{R}^{2048}$ |
| $W_2$ | **Second weight matrix** | $W_2 \in \mathbb{R}^{2048 \times 512}$ (compresses back) |
| $b_2$ | **Second bias** | $b_2 \in \mathbb{R}^{512}$ |
| $\max(0, \cdot)$ | **ReLU activation** | Sets negative values to zero, keeping positive values unchanged |

---

## 3. Visualization Code

```python
"""
Visualization for "Attention Is All You Need"
Vaswani et al. (NIPS 2017)
arXiv: 1706.03762
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches
from scipy.special import softmax


# ============================================================
# 1. Scaled Dot-Product Attention Visualization
# ============================================================
def visualize_scaled_dot_product_attention():
    """Visualize the attention computation: Q, K, V -> attention weights -> output"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Set random seed for reproducibility
    np.random.seed(42)

    # Simulate 3 tokens with d_k = 4
    n_tokens = 3
    d_k = 4

    Q = np.random.randn(n_tokens, d_k)
    K = np.random.randn(n_tokens, d_k)
    V = np.random.randn(n_tokens, d_k)

    # Compute attention
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores, axis=1)
    output = weights @ V

    # ---- Subplot 1: Q, K, V matrices ----
    ax1 = axes[0]

    # Q matrix
    im1 = ax1.imshow(Q, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax1.set_title('Query Matrix Q', fontsize=12, fontweight='bold')
    ax1.set_xlabel(f'd_k = {d_k}')
    ax1.set_ylabel('Sequence Position')
    ax1.set_xticks(range(d_k))
    ax1.set_yticks(range(n_tokens))
    ax1.set_xticklabels([f'dim{i+1}' for i in range(d_k)])
    ax1.set_yticklabels(['I', 'love', 'you'])
    for i in range(n_tokens):
        for j in range(d_k):
            ax1.text(j, i, f'{Q[i, j]:.1f}',
                     ha='center', va='center', color='black' if abs(Q[i, j]) < 0.5 else 'white', fontsize=8)

    # ---- Subplot 2: Attention weights ----
    ax2 = axes[1]

    im2 = ax2.imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Attention Weights\nsoftmax(QK^T / √d_k)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Keys (positions attending to)')
    ax2.set_ylabel('Queries (attending from)')
    ax2.set_xticks(range(n_tokens))
    ax2.set_yticks(range(n_tokens))
    ax2.set_xticklabels(['I', 'love', 'you'])
    ax2.set_yticklabels(['I', 'love', 'you'])

    for i in range(n_tokens):
        for j in range(n_tokens):
            ax2.text(j, i, f'{weights[i, j]:.2f}',
                     ha='center', va='center', color='black' if weights[i, j] < 0.5 else 'white', fontsize=10)

    # ---- Subplot 3: Output ----
    ax3 = axes[2]

    im3 = ax3.imshow(output, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax3.set_title('Output = weights @ V', fontsize=12, fontweight='bold')
    ax3.set_xlabel(f'd_v = {d_k}')
    ax3.set_ylabel('Sequence Position')
    ax3.set_xticks(range(d_k))
    ax3.set_yticks(range(n_tokens))
    ax3.set_xticklabels([f'dim{i+1}' for i in range(d_k)])
    ax3.set_yticklabels(['I', 'love', 'you'])

    for i in range(n_tokens):
        for j in range(d_k):
            ax3.text(j, i, f'{output[i, j]:.1f}',
                     ha='center', va='center', color='black' if abs(output[i, j]) < 0.5 else 'white', fontsize=8)

    plt.suptitle('Scaled Dot-Product Attention: Q, K, V → Attention Weights → Output',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Print the attention pattern interpretation
    print("\n" + "=" * 60)
    print("Attention Pattern Interpretation:")
    print("=" * 60)
    for i, token in enumerate(['I', 'love', 'you']):
        top_attending = np.argsort(weights[i])[::-1][:2]
        top_tokens = ['I', 'love', 'you'][top_attending[0]]
        if len(top_attending) > 1:
            second_tokens = ['I', 'love', 'you'][top_attending[1]]
            print(f"  '{token}' attends most to '{top_tokens}' ({weights[i, top_attending[0]]:.2f}) "
                  f"and '{second_tokens}' ({weights[i, top_attending[1]]:.2f})")
        else:
            print(f"  '{token}' attends most to '{top_tokens}' ({weights[i, top_attending[0]]:.2f})")


# ============================================================
# 2. Multi-Head Attention Visualization
# ============================================================
def visualize_multi_head_attention():
    """Visualize the multi-head attention mechanism with 4 heads"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    np.random.seed(42)

    n_tokens = 4
    d_model = 8
    h = 4
    d_k = d_model // h  # 2

    # Input
    X = np.random.randn(n_tokens, d_model)

    # Different projection matrices for each head (simplified)
    head_names = ['Syntactic', 'Semantic', 'Positional', 'Contextual']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # Compute attention for each head
    all_weights = []
    for head_idx in range(h):
        # Simulate different attention patterns for each head
        np.random.seed(42 + head_idx * 10)
        W_q = np.random.randn(d_model, d_k) * 0.5
        W_k = np.random.randn(d_model, d_k) * 0.5

        Q = X @ W_q
        K = X @ W_k

        # Create different attention patterns for visualization
        if head_idx == 0:  # Syntactic: attend to adjacent tokens
            weights = np.eye(n_tokens) * 0.5 + 0.1
            for i in range(n_tokens):
                if i > 0:
                    weights[i, i-1] = 0.3
                if i < n_tokens - 1:
                    weights[i, i+1] = 0.3
            weights = weights / weights.sum(axis=1, keepdims=True)
        elif head_idx == 1:  # Semantic: attend to "important" tokens
            weights = np.array([
                [0.1, 0.1, 0.4, 0.4],
                [0.1, 0.1, 0.4, 0.4],
                [0.3, 0.3, 0.2, 0.2],
                [0.3, 0.3, 0.2, 0.2]
            ])
        elif head_idx == 2:  # Positional: strong self-attention
            weights = np.eye(n_tokens) * 0.8 + 0.05
            weights = weights / weights.sum(axis=1, keepdims=True)
        else:  # Contextual: attend to first token (like a "CLS" token)
            weights = np.zeros((n_tokens, n_tokens))
            weights[:, 0] = 0.7
            weights = weights + np.eye(n_tokens) * 0.1
            weights = weights / weights.sum(axis=1, keepdims=True)

        all_weights.append(weights)

        # Plot each head's attention pattern
        row = head_idx // 2
        col = head_idx % 2
        ax = axes[row, col]

        im = ax.imshow(weights, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {head_idx+1}: {head_names[head_idx]}', fontsize=11, fontweight='bold', color=colors[head_idx])
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        ax.set_xticklabels(['tok1', 'tok2', 'tok3', 'tok4'])
        ax.set_yticklabels(['tok1', 'tok2', 'tok3', 'tok4'])

        for i in range(n_tokens):
            for j in range(n_tokens):
                ax.text(j, i, f'{weights[i, j]:.2f}',
                        ha='center', va='center', color='black' if weights[i, j] < 0.5 else 'white', fontsize=9)

    # ---- Combined output ----
    ax_combined = axes[1, 2]

    # Concatenate head outputs (simplified)
    combined = np.concatenate([np.random.randn(n_tokens, d_k) + i * 0.5 for i in range(h)], axis=1)
    im_combined = ax_combined.imshow(combined, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax_combined.set_title('Concatenated Output\nConcat(head₁, ..., headₕ) @ W^O', fontsize=11, fontweight='bold')
    ax_combined.set_xlabel(f'd_model = {d_model}')
    ax_combined.set_ylabel('Sequence Position')
    ax_combined.set_xticks(range(d_model))
    ax_combined.set_yticks(range(n_tokens))
    ax_combined.set_xticklabels([f'h{i+1}' for i in range(h) for _ in range(d_k)], rotation=45, fontsize=8)

    plt.suptitle('Multi-Head Attention: h = 4 Heads in Parallel',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Multi-Head Attention Interpretation:")
    print("=" * 60)
    for i, (name, weights) in enumerate(zip(head_names, all_weights)):
        print(f"\nHead {i+1} ({name}):")
        for j in range(n_tokens):
            top = np.argmax(weights[j])
            print(f"  Token {j+1} attends most to Token {top+1} ({weights[j, top]:.2f})")


# ============================================================
# 3. Positional Encoding Visualization
# ============================================================
def visualize_positional_encoding():
    """Visualize the sinusoidal positional encodings"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    d_model = 128
    max_pos = 50

    # Compute positional encodings
    pos = np.arange(max_pos)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]

    # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates

    # Apply sin to even indices and cos to odd indices
    pe = np.zeros((max_pos, d_model))
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # ---- Subplot 1: Heatmap of positional encodings ----
    ax1 = axes[0]
    im1 = ax1.imshow(pe, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title('Positional Encoding Heatmap', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Dimension (d_model)')
    ax1.set_ylabel('Position (pos)')
    ax1.set_xticks([0, 32, 64, 96, 127])
    ax1.set_xticklabels(['0', '32', '64', '96', '127'])
    plt.colorbar(im1, ax=ax1, label='Encoding Value')

    # ---- Subplot 2: First 10 dimensions as line plots ----
    ax2 = axes[1]
    dims_to_plot = [0, 1, 2, 3, 4, 5]
    colors_line = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for dim, color in zip(dims_to_plot, colors_line):
        ax2.plot(range(max_pos), pe[:, dim], color=color, linewidth=2,
                 label=f'dim {dim}' + (' (sin)' if dim % 2 == 0 else ' (cos)'))

    ax2.set_title('Positional Encodings (First 6 Dimensions)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Position (pos)')
    ax2.set_ylabel('Encoding Value')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    # ---- Subplot 3: Relative position property ----
    ax3 = axes[2]

    # Show that PE(pos+k) can be represented as linear function of PE(pos)
    # This is the key insight: sinusoidal encodings allow learning relative positions

    pos1 = 10
    pos2 = 15
    offset = pos2 - pos1

    # Plot the encodings for the two positions
    ax3.plot(range(d_model), pe[pos1, :], 'b-', linewidth=2, label=f'PE({pos1})')
    ax3.plot(range(d_model), pe[pos2, :], 'r-', linewidth=2, label=f'PE({pos2})')

    # Highlight that they are related
    ax3.fill_between(range(d_model), pe[pos1, :], pe[pos2, :], alpha=0.2, color='purple')

    ax3.set_title(f'PE({pos2}) is a Linear Transform of PE({pos1})\nOffset = {offset} positions',
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Dimension')
    ax3.set_ylabel('Encoding Value')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)

    plt.suptitle('Positional Encoding: Adding Sequence Order Information',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Positional Encoding Insight:")
    print("=" * 60)
    print("  Sinusoidal encodings allow the model to learn relative positions.")
    print("  For any fixed offset k, PE(pos+k) can be represented as a linear")
    print("  function of PE(pos), enabling the model to attend by relative position.")
    print(f"  First 5 dimensions at pos=0: {pe[0, :5].round(3)}")
    print(f"  First 5 dimensions at pos=10: {pe[10, :5].round(3)}")


# ============================================================
# 4. Transformer Architecture Overview
# ============================================================
def visualize_transformer_architecture():
    """Visualize the overall Transformer architecture"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ---- Encoder side (left) ----
    # Input
    ax.text(2.5, 9.2, 'Input', fontsize=13, fontweight='bold', ha='center')
    ax.text(2.5, 8.8, '(x₁, x₂, ..., xₙ)', fontsize=10, ha='center', style='italic')

    # Input Embedding
    box = FancyBboxPatch((1.5, 7.5), 2.0, 0.8,
                          boxstyle="round,pad=0.05", edgecolor='#2c3e50',
                          facecolor='#d6eaf8', linewidth=2)
    ax.add_patch(box)
    ax.text(2.5, 7.9, 'Input Embedding', fontsize=11, ha='center', va='center')

    # + Positional Encoding
    ax.annotate('', xy=(2.5, 7.5), xytext=(2.5, 6.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(1.0, 7.0, '+', fontsize=14, fontweight='bold', ha='center')
    ax.text(1.0, 6.6, 'Positional', fontsize=9, ha='center', style='italic')
    ax.text(1.0, 6.3, 'Encoding', fontsize=9, ha='center', style='italic')

    # N=6 Encoder Layers
    for i in range(6):
        y = 6.0 - i * 0.9

        # Layer box
        box = FancyBboxPatch((1.0, y - 0.35), 3.0, 0.7,
                              boxstyle="round,pad=0.05", edgecolor='#2980b9',
                              facecolor='#ebf5fb' if i % 2 == 0 else '#d6eaf8', linewidth=2)
        ax.add_patch(box)

        # Sub-layers
        ax.text(2.5, y + 0.15, 'Multi-Head Self-Attention', fontsize=9, ha='center', va='center')
        ax.text(2.5, y - 0.1, 'Feed Forward', fontsize=9, ha='center', va='center')

        if i < 5:
            ax.annotate('', xy=(2.5, y - 0.35), xytext=(2.5, y - 0.55),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1))

    # Add + & Norm annotations
    ax.text(4.3, 5.2, '+ & Norm', fontsize=8, ha='center', style='italic', color='#666')

    # Encoder label
    ax.text(2.5, 1.0, 'Encoder', fontsize=13, fontweight='bold', ha='center', color='#2980b9')
    ax.text(2.5, 0.6, 'N = 6 layers', fontsize=10, ha='center', style='italic')

    # ---- Decoder side (right) ----
    # Output
    ax.text(11.5, 9.2, 'Output', fontsize=13, fontweight='bold', ha='center')
    ax.text(11.5, 8.8, '(y₁, y₂, ..., yₘ)', fontsize=10, ha='center', style='italic')

    # Output Embedding
    box = FancyBboxPatch((10.5, 7.5), 2.0, 0.8,
                          boxstyle="round,pad=0.05", edgecolor='#2c3e50',
                          facecolor='#fdebd0', linewidth=2)
    ax.add_patch(box)
    ax.text(11.5, 7.9, 'Output Embedding', fontsize=11, ha='center', va='center')

    # + Positional Encoding
    ax.annotate('', xy=(11.5, 7.5), xytext=(11.5, 6.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(10.0, 7.0, '+', fontsize=14, fontweight='bold', ha='center')
    ax.text(10.0, 6.6, 'Positional', fontsize=9, ha='center', style='italic')
    ax.text(10.0, 6.3, 'Encoding', fontsize=9, ha='center', style='italic')

    # N=6 Decoder Layers
    for i in range(6):
        y = 6.0 - i * 0.9

        # Layer box
        box = FancyBboxPatch((10.0, y - 0.35), 3.0, 0.7,
                              boxstyle="round,pad=0.05", edgecolor='#e67e22',
                              facecolor='#fef9e7' if i % 2 == 0 else '#fdebd0', linewidth=2)
        ax.add_patch(box)

        # Sub-layers
        ax.text(11.5, y + 0.15, 'Masked Self-Attention', fontsize=9, ha='center', va='center')
        ax.text(11.5, y - 0.0, 'Encoder-Decoder Attention', fontsize=9, ha='center', va='center')
        ax.text(11.5, y - 0.15, 'Feed Forward', fontsize=9, ha='center', va='center')

        if i < 5:
            ax.annotate('', xy=(11.5, y - 0.35), xytext=(11.5, y - 0.55),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1))

    # Add + & Norm annotations
    ax.text(13.3, 5.2, '+ & Norm', fontsize=8, ha='center', style='italic', color='#666')

    # Decoder label
    ax.text(11.5, 1.0, 'Decoder', fontsize=13, fontweight='bold', ha='center', color='#e67e22')
    ax.text(11.5, 0.6, 'N = 6 layers', fontsize=10, ha='center', style='italic')

    # ---- Connections between encoder and decoder ----
    # Encoder output -> Decoder (Encoder-Decoder Attention)
    ax.annotate('', xy=(10.0, 4.5), xytext=(4.0, 4.5),
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=2.5, linestyle='dashed'))
    ax.text(7.0, 4.9, 'Keys & Values', fontsize=10, ha='center', color='#8e44ad', fontweight='bold')
    ax.text(7.0, 4.5, 'from Encoder', fontsize=9, ha='center', color='#8e44ad', style='italic')

    # ---- Final output layer ----
    # Encoder output
    ax.annotate('', xy=(2.5, 0.8), xytext=(2.5, 1.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Decoder output to final linear + softmax
    ax.annotate('', xy=(11.5, 0.8), xytext=(11.5, 1.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Final linear + softmax
    box = FancyBboxPatch((10.0, -0.3), 3.0, 0.6,
                          boxstyle="round,pad=0.05", edgecolor='#27ae60',
                          facecolor='#d5f5e3', linewidth=2)
    ax.add_patch(box)
    ax.text(11.5, 0.0, 'Linear + Softmax', fontsize=11, ha='center', va='center')

    # Output probabilities
    ax.annotate('', xy=(11.5, -0.3), xytext=(11.5, -0.6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(11.5, -0.9, 'Output Probabilities', fontsize=11, fontweight='bold', ha='center')

    # ---- Title ----
    ax.text(7.0, 9.8, 'Transformer Architecture', fontsize=16, fontweight='bold', ha='center')

    # ---- Legend ----
    legend_elements = [
        mpatches.Patch(facecolor='#d6eaf8', edgecolor='#2980b9', label='Encoder Components'),
        mpatches.Patch(facecolor='#fdebd0', edgecolor='#e67e22', label='Decoder Components'),
        mpatches.Patch(facecolor='#d5f5e3', edgecolor='#27ae60', label='Output Layer'),
        mpatches.Patch(facecolor='none', edgecolor='#8e44ad', linestyle='dashed', label='Encoder-Decoder Attention'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, bbox_to_anchor=(0.0, 0.0))

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Model Performance Visualization
# ============================================================
def visualize_model_performance():
    """Visualize the Transformer's performance compared to other models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Data from Table 2 in the paper
    models = ['ByteNet', 'Deep-Att + PosUnk', 'MoE', 'Transformer\n(base)', 'Transformer\n(big)']
    en_de_bleu = [23.75, 25.0, 26.03, 27.3, 28.4]
    en_fr_bleu = [None, 39.2, 40.56, 38.1, 41.8]

    colors_models = ['#95a5a6', '#7f8c8d', '#5d6d7e', '#3498db', '#2ecc71']

    # ---- EN-DE BLEU ----
    ax1 = axes[0]
    bars1 = ax1.bar(models, en_de_bleu, color=colors_models, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars1, en_de_bleu):
        if val is not None:
            ax1.text(bar.get_x() + bar.get_width()/2., val + 0.3,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('BLEU Score', fontsize=13)
    ax1.set_title('WMT 2014 EN-DE Translation', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 32)
    ax1.grid(axis='y', alpha=0.3)

    # Annotate the improvement
    ax1.annotate('+2.0 BLEU\nover previous best',
                 xy=(4, 28.4), xytext=(3.5, 30.5),
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
                 fontsize=10, color='#2ecc71', ha='center')

    # ---- EN-FR BLEU ----
    ax2 = axes[1]
    # Filter out None values
    models_fr = ['Deep-Att + PosUnk', 'MoE', 'Transformer\n(base)', 'Transformer\n(big)']
    en_fr_bleu_filtered = [39.2, 40.56, 38.1, 41.8]
    colors_fr = ['#7f8c8d', '#5d6d7e', '#3498db', '#2ecc71']

    bars2 = ax2.bar(models_fr, en_fr_bleu_filtered, color=colors_fr, edgecolor='black', linewidth=1.5)

    for bar, val in zip(bars2, en_fr_bleu_filtered):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.3,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('BLEU Score', fontsize=13)
    ax2.set_title('WMT 2014 EN-FR Translation', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 46)
    ax2.grid(axis='y', alpha=0.3)

    # Annotate the improvement
    ax2.annotate('New SOTA!\n41.8 BLEU',
                 xy=(3, 41.8), xytext=(1.5, 43.5),
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2),
                 fontsize=10, color='#2ecc71', ha='center')

    ax2.annotate('3.5 days on 8 GPUs',
                 xy=(3, 41.8), xytext=(3, 44.8),
                 fontsize=9, color='#555', ha='center', style='italic')

    plt.suptitle('Transformer Performance: State-of-the-Art Results',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================
# Execute visualizations
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Attention Is All You Need")
    print("Vaswani et al. (NIPS 2017) | arXiv: 1706.03762")
    print("=" * 70)
    print("\nVisualizing Core Concepts...\n")

    print("1. Scaled Dot-Product Attention...")
    visualize_scaled_dot_product_attention()

    print("\n2. Multi-Head Attention...")
    visualize_multi_head_attention()

    print("\n3. Positional Encoding...")
    visualize_positional_encoding()

    print("\n4. Transformer Architecture...")
    visualize_transformer_architecture()

    print("\n5. Model Performance...")
    visualize_model_performance()

    print("\n" + "=" * 70)
    print("Key Takeaways:")
    print("1. Transformer uses only attention mechanisms (no RNNs or CNNs).")
    print("2. Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V")
    print("3. Multi-Head Attention: h parallel attention heads with different projections.")
    print("4. Positional Encoding: Sinusoidal functions inject sequence order information.")
    print("5. Achieved 28.4 BLEU on EN-DE and 41.8 BLEU on EN-FR translation.")
    print("6. Training is significantly more parallelizable and faster than RNNs.")
    print("=" * 70)
```

### Code Output Interpretation

| Visualization | Content | Key Insight |
|------|---------|---------|
| **Scaled Dot-Product Attention** | Q, K, V matrices → attention weights → output | Shows how each token attends to others with softmax weights |
| **Multi-Head Attention** | 4 parallel heads with different attention patterns | Each head learns different types of relationships (syntactic, semantic, etc.) |
| **Positional Encoding** | Heatmap + line plots of sinusoidal encodings | Encodings allow the model to learn relative positions |
| **Transformer Architecture** | Complete encoder-decoder diagram | N=6 layers with residual connections and layer norm |
| **Model Performance** | BLEU scores on WMT 2014 | Transformer outperforms previous SOTA models |

---

## 4. Academic Sources

- **Original Paper**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NIPS)*, 30, 5998-6008
- **arXiv Preprint**: https://arxiv.org/abs/1706.03762
- **NIPS 2017**: Presented at the 31st Conference on Neural Information Processing Systems
- **Google Research**: All authors were affiliated with Google (Brain or Research) at the time of publication

### Core Contributions Summary

1. **Transformer Architecture**: The first sequence transduction model based entirely on attention mechanisms, dispensing with recurrence and convolutions

2. **Scaled Dot-Product Attention**: Attention function with scaling factor $\sqrt{d_k}$ to prevent large dot products from pushing the softmax into regions with extremely small gradients

3. **Multi-Head Attention**: Projects queries, keys, and values $h$ times with different learned projections, allowing the model to jointly attend to information from different representation subspaces

4. **Positional Encoding**: Sinusoidal functions of different frequencies that inject information about the relative or absolute position of tokens in the sequence

5. **State-of-the-Art Performance**: 28.4 BLEU on WMT 2014 English-to-German and 41.8 BLEU on English-to-French, with significantly less training time

6. **Parallelization**: The Transformer allows for significantly more parallelization than recurrent models, making training much faster

---

Theoretically, self-attention is **slower than RNNs and CNNs**, but in practice, on GPUs, it is **much faster for training**.

## 1. The Theory: Why Attention is Slower

The theoretical computational complexity explains why, in principle, attention is the slowest:

| Layer Type | Complexity Per Layer | Sequential Operations | Maximum Path Length |
| :--- | :--- | :--- | :--- |
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| **Recurrent (RNN)** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional (CNN)** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

- **The Bottleneck: $O(n^2 \cdot d)$**. The $n^2$ term means that as the input sequence length ($n$) grows, the computational cost of self-attention grows **quadratically**. This is because it must compute a similarity score between every pair of tokens in the sequence. This makes it theoretically much slower than RNNs ($O(n)$) or CNNs ($O(k \cdot n)$) for long sequences.

- **Why $O(n)$ from RNNs is a Bottleneck**: While an RNN's complexity is linear, its $O(n)$ *sequential operations* mean it must process tokens one after another. This creates a strict dependency that cannot be parallelized.

## 2. The Practice: Why Transformers are Faster

Despite its higher theoretical complexity, the Transformer is significantly faster in practice for several key reasons:

1.  **Massive Parallelization**: This is the single most important factor. The self-attention mechanism's $O(n^2)$ operations are almost entirely **matrix multiplications**. GPUs are specifically designed to perform these types of operations in parallel across thousands of cores. In contrast, an RNN's sequential operations ($O(n)$) cannot be parallelized because each step depends on the previous one.

2.  **Shorter "Maximum Path Length"**: This measures how many steps information must travel between distant parts of the input. For self-attention, this is $O(1)$, meaning any two tokens can directly interact in a single step. For an RNN, it's $O(n)$, meaning information must traverse the entire sequence step-by-step. This makes Transformers vastly superior at capturing long-range dependencies.

## 3. The Catch: Training vs. Inference

It is crucial to distinguish between training and inference (generation).

- **Training**: The Transformer's parallel nature makes it **5-10x faster** to train than an LSTM. The ability to process the entire sequence at once leads to huge speedups.

- **Inference (Autoregressive Generation)**: During generation, Transformers become slower. They must produce tokens one by one, and for each new token, they re-compute attention over the entire previous sequence. This leads to high memory bandwidth usage and computational cost, making them less efficient for low-latency or low-resource applications compared to RNNs.

### Summary

| Aspect | Self-Attention (Transformer) | RNN |
| :--- | :--- | :--- |
| **Theoretical Complexity** | Slower ($O(n^2)$) | Faster ($O(n)$) |
| **Sequential Operations** | Constant ($O(1)$) - **Highly Parallelizable** | Linear ($O(n)$) - **Sequential Bottleneck** |
| **Training Speed** | **Much Faster** (5-10x) | Slower |
| **Inference Speed** | Slower (due to re-computation) | Faster (constant memory) |
| **Long-Range Dependencies** | **Excellent** (Path length $O(1)$) | Poor (Path length $O(n)$) |

In short, the Transformer trades a higher theoretical complexity for **extreme parallelizability**. This trade-off makes it significantly faster to train on modern hardware, despite being theoretically "slower."
