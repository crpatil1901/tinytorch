import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __neg__(self):
        return self * -1
    
    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only Supports int / float"
        out = Value(self.data ** other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out
    
    def exp(self):
        x = self.data
        e = math.exp(x)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(s, (self,), 'Sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def elu(self, alpha=1.0):
        x = self.data
        e = x if x >= 0 else alpha * (math.exp(x) - 1)
        out = Value(e, (self,), 'ELU')
        def _backward():
            self.grad += out.grad if x >= 0 else alpha * math.exp(x) * out.grad
        out._backward = _backward
        return out

    def step(self):
        out = Value(1 if self.data > 0 else 0, (self,), 'Step')
        def _backward():
            self.grad += 0  # Step function has no gradient
        out._backward = _backward
        return out

    def log(self):
        x = self.data
        assert x > 0, "Logarithm only defined for positive numbers"
        l = math.log(x)
        out = Value(l, (self,), 'Log')
        def _backward():
            self.grad += (1 / x) * out.grad
        out._backward = _backward
        return out

    def loginv(self):
        x = self.data
        exp_x = math.exp(x)
        out = Value(exp_x, (self,), 'LogInv')
        def _backward():
            self.grad += exp_x * out.grad
        out._backward = _backward
        return out

    def sin(self):
        x = self.data
        s = math.sin(x)
        out = Value(s, (self,), 'Sin')
        def _backward():
            self.grad += math.cos(x) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        x = self.data
        c = math.cos(x)
        out = Value(c, (self,), 'Cos')
        def _backward():
            self.grad += -math.sin(x) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order = []
        visited = set()
        def topo_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topo_sort(child)
                order.append(v)
        topo_sort(self)
        self.grad = 1.0
        for node in order[::-1]:
            node._backward()
