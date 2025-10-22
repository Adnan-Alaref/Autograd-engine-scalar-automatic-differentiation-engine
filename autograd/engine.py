import torch
import math
import numbers

class Value:
  '''shared across all instances use in train/eval mode'''
  grad_enabled = True

  '''save memory and speed up attribute access than __dict__.'''
  __slots__ = ('data', '_prev', '_op', 'grad', '_backward', 'label')

  def __init__(self, data, _children=() , _op='', label=''):

    # --- Auto-cast logic ---
    if isinstance(data, torch.Tensor):
      data = data.item() # extract float from 0-dim tensor
    elif isinstance(data,(list,tuple)):
       raise TypeError("Value class supports only scalar data, not lists or arrays.")
    elif not isinstance(data, numbers.Real):
      raise TypeError(f"Unsupported data type {type(data)} for Value.")

    self.data = float(data)
    self.grad = 0.0
    self._op = _op
    self.label = label
    self._prev = set(_children)
    self._backward = lambda:None

  # --- Basic arithmetic operations ---
  '''Support Basic arithmetic operations either Value or a number'''

  # Addition
  def __add__(self, other):
    is_other_value = isinstance(other, Value)
    if is_other_value:
      out = Value(self.data + other.data, (self,other), '+')
    else:
      out = Value(self.data + other, (self,), '+')

    if Value.grad_enabled:
      def _backward():
        self.grad += 1.0 * out.grad
        if is_other_value:
          other.grad += 1.0 * out.grad
      out._backward = _backward
    return out

  def __radd__(self, other):
    # 1 + x → int.__add__(1, x) → (fails) → x.__radd__(1)
    return self.__add__(other)

  # Subtraction
  def __sub__(self, other):
    is_other_value = isinstance(other, Value)
    if is_other_value:
      out = Value(self.data - other.data, (self, other), '-')
    else:
      out = Value(self.data - other, (self,), '-')

    if Value.grad_enabled:
      def _backward():
          self.grad += 1.0 * out.grad
          if is_other_value:
            other.grad += -1.0 * out.grad
      out._backward = _backward
    return out

  def __rsub__(self, other):
    # 1 - x → int.__sub__(1,x) → (fails) → x.__rsub__(1)
    is_other_value = isinstance(other, Value)
    if is_other_value:
      out = Value(other.data - self.data, (other, self), '-')
    else:
      out = Value(other - self.data, (self,), '-')

    if Value.grad_enabled:
      def _backward():
        self.grad += -1.0 * out.grad
        if is_other_value:
          other.grad += 1.0 * out.grad
      out._backward = _backward
    return out

  # Multiplication
  def __mul__(self, other):
    is_other_value = isinstance(other, Value)
    if is_other_value:
      out = Value(self.data * other.data, (self,other), '*')
    else:
      out = Value(self.data * other, (self,), '*')

    if Value.grad_enabled:
      def _backward():
        self.grad += (other.data if is_other_value else other) * out.grad
        if is_other_value:
          other.grad += self.data * out.grad
      out._backward = _backward
    return out

  def __rmul__(self, other):
    # 1*x → int.__mul__(1,x) → (fails) → x.__rmul__(1)
    return self.__mul__(other)

  # True Division
  def __truediv__(self, other):
    # a/b = a*b**-1
    is_other_value = isinstance(other, Value)
    if is_other_value:
      if other.data == 0:
          raise ZeroDivisionError("division by zero")
      out = Value(self.data * (other.data**-1), (self, other), '/')

    else:
      if other == 0:
            raise ZeroDivisionError("division by zero")
      out = Value(self.data * (other**-1), (self,), '/')

    if Value.grad_enabled:
      def _backward():
        if is_other_value:
          # numerical regularization or epsilon stabilization
          safe_other = other.data if abs(other.data) > 1e-12 else 1e-12

          self.grad += (1 / safe_other) * out.grad
          other.grad += ((-self.data)/(safe_other**2)) * out.grad

        else:
          # a / c = d(a/c)/da = 1/c
          self.grad +=(1/other) * out.grad
      out._backward = _backward
    return out

  def __rtruediv__(self,other):
    # 1 / x → int.__truediv__(1, x) → (fails) → x.__rtruediv__(1)
    return Value(other) * (self**-1)

  # Power
  def __pow__(self, other):
    is_other_value = isinstance(other, Value)
    if is_other_value:
      out = Value(self.data** other.data, (self,other), f'**{other}')

    else:
      out = Value(self.data** other, (self,), f'**{other}')

    if Value.grad_enabled:
      def _backward():
        if is_other_value:
          # 1. Update gradient for the base (self) ∂f/∂x = y * x^(y-1)
          self.grad += other.data * (self.data**(other.data-1)) * out.grad

          # 2. Update gradient for the exponent (other) ∂f/∂y = x^y * ln(x)
          '''Use safe base to avoid log(0) or negative issues'''
          base = self.data if self.data > 0 else 1e-8
          other.grad += (self.data**other.data) * math.log(base) * out.grad

        else:
          # When other is constant: d(x^a)/dx = a * x^(a-1)
          self.grad += other * (self.data**(other-1)) * out.grad
      out._backward = _backward
    return out

  def __rpow__(self, other):
    # 1**x → int.__pow__(1,x) → (fails) → x.__rpow__(1)
    base = other
    out = Value(base**self.data, (self,), f'**{base}' )

    if Value.grad_enabled:
      def _backward():
        safe_base = base if base > 0 else 1e-8
        self.grad += (base**self.data) * math.log(safe_base) * out.grad
      out._backward = _backward
    return out

  # Ngative numbers
  def __neg__(self):
    out = Value(self.data *-1, (self,) ,'neg')

    if Value.grad_enabled:
      def _backward():
        self.grad += -1 * out.grad
      out._backward = _backward
    return out

  # --- Activation functions (non linearity)---

  # Tanh
  def tanh(self):
    x = self.data
    t_val = math.tanh(x)
    # t_val = (math.exp(2*x) -1)/(math.exp(2*x) +1)
    out = Value(t_val, (self,), 'Tanh')

    if Value.grad_enabled:
      def _backward():
        self.grad += (1 - t_val**2) * out.grad
      out._backward = _backward
    return out

  # Rule
  def relu(self):
    r_val = self.data if self.data > 0 else 0.0
    out = Value(r_val, (self,), 'RelU')

    if Value.grad_enabled:
      def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0)  * out.grad
      out._backward = _backward
    return out

  # Sigmoid
  def sigmoid(self):
    s_val = 1 / (1+math.exp(-self.data))
    out = Value(s_val, (self,), 'Sigmoid')

    if Value.grad_enabled:
      def _backward():
        self.grad += s_val * (1-s_val) * out.grad
      out._backward = _backward
    return out

  # Exponential
  def exp(self):
    e_val = math.exp(min(self.data, 50))
    out = Value(e_val, (self,), 'exp')

    if Value.grad_enabled:
      def _backward():
        self.grad += e_val * out.grad
      out._backward = _backward
    return out

  # Log
  def log(self):
    if self.data <= 0:
      raise ValueError("Cannot calculate the natural log (ln) for non-positive numbers (x <= 0).")

    log_val = math.log(self.data)
    out = Value(log_val, (self,), 'log')

    if Value.grad_enabled:
      def _backward():
        self.grad += (1 / self.data) * out.grad
      out._backward = _backward
    return out

  # Absolute
  def abs(self):
    abs_val = self.data if self.data >= 0 else -self.data
    out = Value(abs_val, (self,), 'abs')

    if Value.grad_enabled:
      def _backward():
        self.grad +=(1 if self.data>0 else -1 if self.data < 0 else 0) *out.grad
      out._backward = _backward
    return out
  
  # Backpropagation
  def backward(self):
    stack = []
    visted = set()

    def dfs(root):
      if root in visted:
        return
      visted.add(root)
      for _child in root._prev:
        dfs(_child)
      stack.append(root)

    # Build topological order
    dfs(self)

    # Backward pass
    self.grad = 1.0
    for node in reversed(stack):
      node._backward()

  def to_tensor(self):
    return torch.tensor(self.data, dtype=torch.float32, requires_grad=False)

  def __repr__(self) -> str:
    return f"Value(data={self.data})"