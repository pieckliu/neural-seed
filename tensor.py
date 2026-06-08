import numpy as np

# ============================================================
# 1. Autograd Tensor
# ============================================================

def unbroadcast(grad, shape):
    """
    把 broadcast 之后的梯度还原回原始 tensor 的 shape。
    """
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+"
        )

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*"
        )

        def _backward():
            if self.requires_grad:
                self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power):
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{power}"
        )

        def _backward():
            if self.requires_grad:
                self.grad += power * (self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@"
        )

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum"
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad

                if axis is not None and not keepdims:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)

                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)

                self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            axes = axis if isinstance(axis, tuple) else (axis,)
            denom = np.prod([self.data.shape[a] for a in axes])

        return self.sum(axis=axis, keepdims=keepdims) / denom

    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu"
        )

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)

        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh"
        )

        def _backward():
            if self.requires_grad:
                self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        从当前 scalar loss 节点开始，反向传播整个计算图。
        """
        if self.data.shape != ():
            raise RuntimeError("backward() currently requires a scalar loss.")

        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
