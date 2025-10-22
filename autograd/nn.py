import random
from .engine import Value
from itertools import chain

class Module:
  __slots__ = ('lr', 'best_loss', 'training', 'patience')
  def __init__(self):
    self.lr = 0.1
    self.patience = 0
    self.training = True
    self.best_loss = float('inf')

  def parameters(self):
    return []

  def children(self):
    return []

  def train(self):
    self.training = True
    Value.grad_enabled = True
    for _child in self.children():
      _child.train()

  def eval(self):
    self.training = False
    Value.grad_enabled = False
    for _child in self.children():
      _child.eval()

  def zero_grad(self):
    for p in self.parameters():
      p.grad = 0.0 if hasattr(p,'grad') else None

  def update(self, loss:float):
    # --- Update parameters ---
    for p in self.parameters():
      p.data -= self.lr * p.grad

    # --- Dynamic LR adjustment ---
    if loss.data < self.best_loss:
      self.best_loss = loss.data
      self.patience = 0
      self.lr *=1.02 # small increase if improving

    else:
      self.patience +=1
      if self.patience >= 3: # wait 3 bad steps before reducing
        self.lr *=0.8
        self.patience = 0

    # clamp LR to avoid explosion
    self.lr = max(min(self.lr,1.0),1e-4)

class Neuron(Module):
  __slots__ = ('weights','bias', "act_func")

  def __init__(self,nin:int, act_func="relu"):
    super().__init__()
    self.act_func = act_func
    self.bias = Value(random.uniform(-1, 1), label='b')
    self.weights = [Value(random.uniform(-1, 1),label=f'w{k}') for k in range(nin)]

  def __call__(self, x):
    return self.activate(x)

  def forward(self,x:list['Value'])->Value:
    # wi * xi + b
    net = sum((wi*xi for (wi,xi)in zip(self.weights, x)), start=self.bias)
    return net

  def activate(self, x):
    out = self.forward(x)
    if self.act_func==None:
      return out # row output

    activations = {
        "tanh":out.tanh,
        "relu":out.relu,
        "sigmoid":out.sigmoid
    }
    if self.act_func not in activations:
      raise ValueError(f"Unknown activation function: {self.act_func}")
    return activations[self.act_func]()

  def parameters(self):
    """Return all trainable parameters in neuron"""
    return self.weights + [self.bias]

class Layer(Module):
  __slots__ = ("nin", "nout", "neurons", "act_func")

  def __init__(self, nin, nout, act_func="relu"):
    super().__init__()
    self.nin = nin
    self.nout = nout
    self.act_func = act_func
    self.neurons = [Neuron(self.nin, act_func=self.act_func) for _ in range(self.nout)]

  def __call__(self,x):
    return self.forward(x)

  def children(self):
    return self.neurons

  def forward(self, x:list['Value'])->Value:
    nets = [neuron(x) for neuron in self.neurons]
    return nets[0] if len(nets)==1 else nets

  def parameters(self):
    """Return all trainable parameters from all neurons in the layer"""
    return list(chain.from_iterable(n.parameters() for n in self.neurons))

class MLP(Module):
  __slots__ = ("nin", "nouts", "sz", "layers","act_func")

  def __init__(self, nin:int, nouts:list, act_func="relu"):
    super().__init__()
    self.sz = [nin] + nouts
    self.act_func = act_func
    self.layers = [Layer(nin=self.sz[i], nout=self.sz[i+1],
                      act_func=(self.act_func if i<len(nouts)-1 else None)) for i in range(len(nouts))]

  def __call__(self,x):
    return self.forward(x)

  def children(self):
    return self.layers

  def forward(self,x:list['Value'])->Value:
    for layer in self.layers:
      x = layer(x) # feed output of one layer to the next
    return x

  def parameters(self):
    return list(chain.from_iterable(layer.parameters() for layer in self.layers))

  def summary(self):
    print("=================================================")
    print(f"{'Layer (type)':20} {'Output Shape':20} {'Param #':10}")
    print("=================================================")
    total_params = 0

    for i, layer in enumerate(self.layers):
        layer_name = layer.__class__.__name__
        out_shape = f"[{layer.nout}]"
        param_count = len(layer.parameters())
        total_params += param_count
        print(f"Layer-{i+1} ({layer.nin:2d} → {layer.nout:2d})".ljust(25) +\
              f"{out_shape:<20}{param_count:<10}")

    print("=================================================")
    print(f"Total params: {total_params}")
    print("=================================================")