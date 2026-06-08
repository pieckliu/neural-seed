import numpy as np
from tensor import Tensor, unbroadcast

# ============================================================
# 2. Model components
# ============================================================

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = Tensor(
            np.random.randn(in_dim, out_dim) * np.sqrt(2 / in_dim),
            requires_grad=True
        )

        self.b = Tensor(
            np.zeros((1, out_dim)),
            requires_grad=True
        )

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    def __init__(self, sizes, activation="relu"):
        """
        sizes 例如：
        [2, 32, 32, 1]
        """
        self.layers = [
            Linear(sizes[i], sizes[i + 1])
            for i in range(len(sizes) - 1)
        ]

        self.activation = activation

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)

            if self.activation == "relu":
                x = x.relu()
            elif self.activation == "tanh":
                x = x.tanh()
            elif self.activation == "none":
                pass
            else:
                raise ValueError(f"Unknown activation: {self.activation}")

        return self.layers[-1](x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

