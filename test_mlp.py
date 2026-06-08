import numpy as np
import matplotlib.pyplot as plt

from tensor import Tensor, unbroadcast
from model.mlp import MLP, Linear, mse_loss, make_moons_np, accuracy_from_logits, train_model


# ============================================================
# 3. Data and loss
# ============================================================

def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()


def make_moons_np(n_samples=400, noise=0.15, random_seed=0):
    """
    手写 two moons 数据。
    """
    rng = np.random.default_rng(random_seed)

    n1 = n_samples // 2
    n2 = n_samples - n1

    theta1 = rng.uniform(0, np.pi, size=n1)
    x1 = np.stack([
        np.cos(theta1),
        np.sin(theta1)
    ], axis=1)

    theta2 = rng.uniform(0, np.pi, size=n2)
    x2 = np.stack([
        1.0 - np.cos(theta2),
        -np.sin(theta2) + 0.5
    ], axis=1)

    X = np.concatenate([x1, x2], axis=0)

    y = np.concatenate([
        np.zeros((n1, 1)),
        np.ones((n2, 1))
    ], axis=0)

    X += rng.normal(scale=noise, size=X.shape)

    idx = rng.permutation(n_samples)
    X = X[idx]
    y = y[idx]

    return X.astype(np.float64), y.astype(np.float64)


def accuracy_from_logits(logits, y):
    pred = (logits > 0.5).astype(np.float64)
    return (pred == y).mean()


def train_model(model, X_np, y_np, lr=0.03, epochs=5000, print_every=500):
    for epoch in range(epochs + 1):
        X = Tensor(X_np)
        y = Tensor(y_np)

        pred = model(X)
        loss = mse_loss(pred, y)

        for p in model.parameters():
            p.zero_grad()

        loss.backward()

        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % print_every == 0:
            acc = accuracy_from_logits(pred.data, y_np)
            print(f"epoch={epoch:5d}, loss={loss.data.item():.6f}, acc={acc:.4f}")

    final_pred = model(Tensor(X_np)).data
    final_acc = accuracy_from_logits(final_pred, y_np)
    print(f"Final accuracy: {final_acc:.4f}")



# ============================================================
# 8.5 Forward helper: manually get hidden activations
# ============================================================

def get_hidden_states_numpy(model, X_np):
    """
    手动计算模型每一层的 pre-activation 和 activation。

    对于 model = MLP([2, 32, 32, 1]):

    z1 = X W1 + b1
    h1 = ReLU(z1)

    z2 = h1 W2 + b2
    h2 = ReLU(z2)

    out = h2 W3 + b3
    """
    # layer 1
    W1 = model.layers[0].W.data
    b1 = model.layers[0].b.data
    z1 = X_np @ W1 + b1
    h1 = np.maximum(0, z1)

    # layer 2
    W2 = model.layers[1].W.data
    b2 = model.layers[1].b.data
    z2 = h1 @ W2 + b2
    h2 = np.maximum(0, z2)

    # output layer
    W3 = model.layers[2].W.data
    b3 = model.layers[2].b.data
    out = h2 @ W3 + b3

    return {
        "z1": z1,
        "h1": h1,
        "z2": z2,
        "h2": h2,
        "out": out
    }

def plot_single_second_layer_relu_gate(model, X, y, neuron_id=0, resolution=300):
    """
    画第二层某一个 ReLU 神经元的开关区域。

    第二层第 j 个神经元：

        z2_j = h1 @ W2[:, j] + b2[j]

    其中：

        h1 = ReLU(X @ W1 + b1)

    所以 z2_j = 0 投影回原始二维输入空间后，
    通常是一条 piecewise-linear 的折线边界，而不是单条直线。
    """
    if len(model.layers) < 3:
        raise ValueError("This function expects a model with at least 2 hidden layers + output layer.")

    W2 = model.layers[1].W.data

    if neuron_id < 0 or neuron_id >= W2.shape[1]:
        raise ValueError(f"neuron_id must be between 0 and {W2.shape[1] - 1}")

    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(
        X,
        resolution=resolution
    )

    states = get_hidden_states_numpy(model, grid)

    z2_j = states["z2"][:, neuron_id].reshape(xx.shape)
    active = (z2_j > 0).astype(float)

    plt.figure(figsize=(7, 6))

    # 第二层 ReLU 是否激活的区域
    plt.contourf(xx, yy, active, alpha=0.30)

    # 第二层该神经元的开关边界 z2_j = 0
    plt.contour(xx, yy, z2_j, levels=[0], linewidths=3)

    # 数据点
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(f"Second-layer ReLU gate: neuron {neuron_id}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def plot_second_layer_relu_contours(model, X, y, max_neurons=None, resolution=300):
    """
    画第二层 ReLU 神经元的开关边界。

    每一条 contour 对应一个第二层神经元：

        z2_j = 0

    和第一层不同，第二层的 z2_j 是第一层 ReLU 后的函数，
    所以投影回输入空间通常是 piecewise-linear curve。
    """
    if len(model.layers) < 3:
        raise ValueError("This function expects a model with at least 2 hidden layers + output layer.")

    W2 = model.layers[1].W.data
    num_neurons = W2.shape[1]

    if max_neurons is None:
        max_neurons = num_neurons
    else:
        max_neurons = min(max_neurons, num_neurons)

    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(
        X,
        resolution=resolution
    )

    states = get_hidden_states_numpy(model, grid)
    z2 = states["z2"]

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    for j in range(max_neurons):
        z2_j = z2[:, j].reshape(xx.shape)

        # 只画 z2_j = 0 的 contour
        # 有些神经元可能在整个可视化区域内都不穿过 0，这种情况下 contour 会画不出来
        try:
            plt.contour(xx, yy, z2_j, levels=[0], linewidths=1.5, alpha=0.55)
        except Exception:
            pass

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(f"Second-layer ReLU switching contours: first {max_neurons} neurons")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()



def plot_activation_regions_second_layer(model, X, y, resolution=300):
    """
    画第二层 activation regions。

    颜色不同表示：
    第二层 32 个 ReLU 的 on/off pattern 不同。

    注意：
    这些区域是第二层在原始输入空间上的投影，
    所以它们的边界通常不是简单直线，而是分段线性边界。
    """
    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(
        X,
        resolution=resolution
    )

    states = get_hidden_states_numpy(model, grid)

    z2 = states["z2"]
    mask2 = (z2 > 0).astype(np.int8)

    unique_patterns, compact_ids = np.unique(
        mask2,
        axis=0,
        return_inverse=True
    )

    Z = compact_ids.reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    plt.contourf(xx, yy, Z, levels=50, alpha=0.35)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(f"Second-layer activation regions: {len(unique_patterns)} regions")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


def plot_activation_regions_all_hidden_layers(model, X, y, resolution=300):
    """
    画第一层 + 第二层共同决定的 activation regions。

    对于两层 hidden ReLU MLP：

        h1 = ReLU(z1)
        h2 = ReLU(z2)

    完整网络的局部线性区域由：
        mask1 = z1 > 0
        mask2 = z2 > 0

    共同决定。

    所以这里把 mask1 和 mask2 拼起来，
    每一种不同的 combined pattern 对应一个局部线性区域。
    """
    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(
        X,
        resolution=resolution
    )

    states = get_hidden_states_numpy(model, grid)

    mask1 = (states["z1"] > 0).astype(np.int8)
    mask2 = (states["z2"] > 0).astype(np.int8)

    combined_mask = np.concatenate([mask1, mask2], axis=1)

    unique_patterns, compact_ids = np.unique(
        combined_mask,
        axis=0,
        return_inverse=True
    )

    Z = compact_ids.reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    plt.contourf(xx, yy, Z, levels=80, alpha=0.35)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(f"All hidden ReLU activation regions: {len(unique_patterns)} regions")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
def get_plot_range(X, margin=0.5):
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    return x_min, x_max, y_min, y_max


def make_grid(X, resolution=300, margin=0.5):
    x_min, x_max, y_min, y_max = get_plot_range(X, margin=margin)

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid, x_min, x_max, y_min, y_max


def plot_data(X, y, title="Two moons data"):
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# ============================================================
# 5. Visualization 1: decision boundary
# ============================================================

def plot_decision_boundary_with_contour(model, X, y, title="Decision boundary"):
    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(X)

    logits = model(Tensor(grid)).data.reshape(xx.shape)
    Z = (logits > 0.5).astype(float)

    plt.figure(figsize=(7, 6))

    plt.contourf(xx, yy, Z, alpha=0.35)
    plt.contour(xx, yy, logits, levels=[0.5], linewidths=3)

    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# ============================================================
# 6. Visualization 2: single ReLU gate
# ============================================================

def plot_single_relu_gate(model, X, y, neuron_id=0):
    """
    画第一层某一个 ReLU 神经元的开关区域。

    h_j(x) = ReLU(w_j^T x + b_j)

    开关边界：
    w_j^T x + b_j = 0
    """
    W = model.layers[0].W.data
    b = model.layers[0].b.data

    if neuron_id < 0 or neuron_id >= W.shape[1]:
        raise ValueError(f"neuron_id must be between 0 and {W.shape[1] - 1}")

    w1 = W[0, neuron_id]
    w2 = W[1, neuron_id]
    bj = b[0, neuron_id]

    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(X)

    z = w1 * xx + w2 * yy + bj
    active = (z > 0).astype(float)

    plt.figure(figsize=(7, 6))

    plt.contourf(xx, yy, active, alpha=0.30)
    plt.contour(xx, yy, z, levels=[0], linewidths=3)

    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(f"Single ReLU gate: first-layer neuron {neuron_id}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# ============================================================
# 7. Visualization 3: all first-layer ReLU switching lines
# ============================================================

def plot_first_layer_relu_lines(model, X, y):
    """
    画第一层所有 ReLU 神经元的切割线。

    每个神经元对应一条：
    w_j^T x + b_j = 0
    """
    W = model.layers[0].W.data
    b = model.layers[0].b.data

    x_min, x_max, y_min, y_max = get_plot_range(X)
    xs = np.linspace(x_min, x_max, 300)

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    for j in range(W.shape[1]):
        w1 = W[0, j]
        w2 = W[1, j]
        bj = b[0, j]

        if abs(w2) > 1e-8:
            ys = -(w1 * xs + bj) / w2
            plt.plot(xs, ys, alpha=0.35)
        elif abs(w1) > 1e-8:
            x_line = -bj / w1
            plt.axvline(x_line, alpha=0.35)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title("First-layer ReLU switching lines")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# ============================================================
# 8. Visualization 4: first-layer activation regions
# ============================================================

def plot_activation_regions_first_layer(model, X, y, resolution=300):
    """
    画第一层 activation regions。

    颜色不同表示：
    第一层 32 个 ReLU 的 on/off pattern 不同。

    在同一个 region 里，第一层 ReLU 的开关模式固定。
    因此在这个 region 内，后续网络局部上是线性函数。
    """
    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(X, resolution=resolution)

    W = model.layers[0].W.data
    b = model.layers[0].b.data

    preact = grid @ W + b
    mask = (preact > 0).astype(np.int8)

    # 用 np.unique 给每一种 activation pattern 编号
    unique_patterns, compact_ids = np.unique(mask, axis=0, return_inverse=True)
    Z = compact_ids.reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    plt.contourf(xx, yy, Z, levels=50, alpha=0.35)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=35)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(f"First-layer activation regions: {len(unique_patterns)} regions")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# ============================================================
# 9. Optional: combined 2x2 visualization
# ============================================================
def plot_first_regions_with_second_layer_contour(model, X, y, neuron_id=0, resolution=300):
    """
    背景：第一层 activation regions
    前景：第二层某个 ReLU 神经元的 z2_j = 0 contour

    这张图最适合观察：
    第二层 ReLU 边界如何穿过第一层切好的区域。
    """
    xx, yy, grid, x_min, x_max, y_min, y_max = make_grid(
        X,
        resolution=resolution
    )

    states = get_hidden_states_numpy(model, grid)

    # 第一层 activation pattern
    mask1 = (states["z1"] > 0).astype(np.int8)
    unique_patterns, compact_ids = np.unique(
        mask1,
        axis=0,
        return_inverse=True
    )
    Z_region = compact_ids.reshape(xx.shape)

    # 第二层第 neuron_id 个神经元的 pre-activation
    z2_j = states["z2"][:, neuron_id].reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    # 背景：第一层 region
    plt.contourf(xx, yy, Z_region, levels=50, alpha=0.30)

    # 前景：第二层 ReLU 的开关边界
    plt.contour(
        xx,
        yy,
        z2_j,
        levels=[0],
        linewidths=3
    )

    # 数据点
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y[:, 0],
        edgecolors="k",
        s=35
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.title(
        f"First-layer regions + second-layer ReLU contour: neuron {neuron_id}"
    )
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    
def plot_four_panel_summary(linear_model, relu_model, X, y, neuron_id=0):
    """
    四联图：
    1. 线性模型边界
    2. 单个 ReLU 开关区域
    3. 第一层所有 ReLU 切割线
    4. 最终 ReLU MLP 决策边界
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    x_min, x_max, y_min, y_max = get_plot_range(X)
    xx, yy, grid, _, _, _, _ = make_grid(X)

    # --------------------------------------------------------
    # Panel 1: Linear decision boundary
    # --------------------------------------------------------
    logits_linear = linear_model(Tensor(grid)).data.reshape(xx.shape)
    Z_linear = (logits_linear > 0.5).astype(float)

    ax = axes[0, 0]
    ax.contourf(xx, yy, Z_linear, alpha=0.35)
    ax.contour(xx, yy, logits_linear, levels=[0.5], linewidths=3)
    ax.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=25)
    ax.set_title("1. Linear model boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --------------------------------------------------------
    # Panel 2: Single ReLU gate
    # --------------------------------------------------------
    W = relu_model.layers[0].W.data
    b = relu_model.layers[0].b.data

    w1 = W[0, neuron_id]
    w2 = W[1, neuron_id]
    bj = b[0, neuron_id]

    z = w1 * xx + w2 * yy + bj
    active = (z > 0).astype(float)

    ax = axes[0, 1]
    ax.contourf(xx, yy, active, alpha=0.35)
    ax.contour(xx, yy, z, levels=[0], linewidths=3)
    ax.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=25)
    ax.set_title(f"2. Single ReLU gate: neuron {neuron_id}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --------------------------------------------------------
    # Panel 3: First-layer ReLU switching lines
    # --------------------------------------------------------
    ax = axes[1, 0]
    ax.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=25)

    xs = np.linspace(x_min, x_max, 300)

    for j in range(W.shape[1]):
        w1 = W[0, j]
        w2 = W[1, j]
        bj = b[0, j]

        if abs(w2) > 1e-8:
            ys = -(w1 * xs + bj) / w2
            ax.plot(xs, ys, alpha=0.35)
        elif abs(w1) > 1e-8:
            x_line = -bj / w1
            ax.axvline(x_line, alpha=0.35)

    ax.set_title("3. First-layer ReLU switching lines")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --------------------------------------------------------
    # Panel 4: Final ReLU MLP decision boundary
    # --------------------------------------------------------
    logits_relu = relu_model(Tensor(grid)).data.reshape(xx.shape)
    Z_relu = (logits_relu > 0.5).astype(float)

    ax = axes[1, 1]
    ax.contourf(xx, yy, Z_relu, alpha=0.35)
    ax.contour(xx, yy, logits_relu, levels=[0.5], linewidths=3)
    ax.scatter(X[:, 0], X[:, 1], c=y[:, 0], edgecolors="k", s=25)
    ax.set_title("4. Final ReLU MLP boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    # --------------------------------------------------------
    # Generate data
    # --------------------------------------------------------
    X_np, y_np = make_moons_np(
        n_samples=400,
        noise=0.15,
        random_seed=0
    )

    plot_data(X_np, y_np, title="Two moons data")

    # --------------------------------------------------------
    # Train linear model
    # --------------------------------------------------------
    print("\nTraining linear model...")
    linear_model = MLP([2, 1], activation="none")

    train_model(
        model=linear_model,
        X_np=X_np,
        y_np=y_np,
        lr=0.03,
        epochs=3000,
        print_every=500
    )

    plot_decision_boundary_with_contour(
        linear_model,
        X_np,
        y_np,
        title="Linear model on two moons"
    )

    # --------------------------------------------------------
    # Train ReLU MLP
    # --------------------------------------------------------
    print("\nTraining ReLU MLP...")
    relu_model = MLP([2, 2, 2, 1], activation="relu")

    train_model(
        model=relu_model,
        X_np=X_np,
        y_np=y_np,
        lr=0.03,
        epochs=5000,
        print_every=500
    )

    # --------------------------------------------------------
    # Visualization sequence
    # --------------------------------------------------------

    # 1. 单个 ReLU 神经元的开关区域
    plot_single_relu_gate(
        relu_model,
        X_np,
        y_np,
        neuron_id=0
    )

    # 2. 第一层所有 ReLU 神经元的切割线
    plot_first_layer_relu_lines(
        relu_model,
        X_np,
        y_np
    )

    # 3. 第一层 activation regions
    plot_activation_regions_first_layer(
        relu_model,
        X_np,
        y_np,
        resolution=300
    )

    # 4. 最终 MLP 决策边界
    plot_decision_boundary_with_contour(
        relu_model,
        X_np,
        y_np,
        title="ReLU MLP on two moons"
    )

    plot_first_regions_with_second_layer_contour(
        relu_model,
        X_np,
        y_np,
        neuron_id=0,
        resolution=300
    )
    
    # 5. 四联图总结
    plot_four_panel_summary(
        linear_model=linear_model,
        relu_model=relu_model,
        X=X_np,
        y=y_np,
        neuron_id=0
    )
    
    # --------------------------------------------------------
    # Second-layer ReLU visualization
    # --------------------------------------------------------

    # 1. 第二层单个 ReLU 神经元的开关区域
    plot_single_second_layer_relu_gate(
        relu_model,
        X_np,
        y_np,
        neuron_id=0,
        resolution=300
    )

    # 2. 第二层所有 ReLU 神经元的开关边界
    plot_second_layer_relu_contours(
        relu_model,
        X_np,
        y_np,
        max_neurons=32,
        resolution=300
    )

    # 3. 第二层 activation regions
    plot_activation_regions_second_layer(
        relu_model,
        X_np,
        y_np,
        resolution=300
    )

    # 4. 第一层 + 第二层共同决定的完整局部线性区域
    plot_activation_regions_all_hidden_layers(
        relu_model,
        X_np,
        y_np,
        resolution=300
    )
    