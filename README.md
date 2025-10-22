# AutoGrad Engine 🚀

A lightweight, educational, and fully functional automatic differentiation engine built from scratch using pure Python — **inspired by the design philosophy of micrograd by Andrej Karpathy**, but **extended with Torch-like features, dynamic learning rate, and graph visualization.**

Unlike micrograd, this implementation introduces PyTorch-inspired design patterns, including a Module hierarchy, dynamic forward graph construction, integrated activation functions, and adaptive learning rate logic.
It bridges the gap between simplicity and modern deep learning framework structure — giving learners both transparency and realistic functionality.

---

## 🧠 Overview

This repository implements a minimal deep learning core framework that includes:

* **Custom scalar-based automatic differentiation engine**
* **Neural network building blocks (Neuron, Layer, MLP)**
* **Dynamic learning rate scheduling during training**
* **Graphviz-based visualization of the computational graph**

The code is intentionally **simple, readable, and modular** — making it perfect for understanding the internal mechanics of backpropagation and gradient descent.

---

## 📂 Project Structure

```
autoGrad/
│
├── engine.py        # Core Value and Module classes
├── nn.py            # Module, Neuron, Layer, and MLP definitions
├── build_graph.py   # Graphviz-based computational graph rendering
├── examples/
│   └── demo.ipynb            # Example training script
│   └── demo_draw_graph.ipynb # Example build computational graph
├── graph_output.png # Example visualization
└── README.md
```

---

## ⚙️ Core Components

### **1. `Value` (engine.py)**

The `Value` class represents a single scalar value in the computational graph.  
It supports **arithmetic operations**, **activation functions**, and **automatic differentiation** through a dynamically constructed graph.

#### Key Features

* **Operator Overloading**: Supports +, -, *, /, **, negation.  
* **Nonlinearities**: `tanh()`, `relu()`, `sigmoid()`, `exp()`, `log()`, `abs()`.  
* **Backpropagation**: Uses a DFS-based topological traversal for gradient computation.  
* **Autocasting**: Accepts Python floats or 0D torch tensors.  
* **Stability Enhancements**: Safe handling for division, log, and exponentiation edge cases.

#### Example

```python
from autograd.engine import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a.tanh()
c.backward()

print(c.data)  # Forward result
print(a.grad, b.grad)  # Computed gradients
```

---

### **2. `Module`, `Neuron`, `Layer`, and `MLP` (nn.py)**

These classes provide a minimal neural network framework built entirely on top of the `Value` engine.

#### **Module**

Base class for all trainable models.  
Includes:

* Learning rate and patience tracking  
* Dynamic LR adjustment based on performance  
* Unified `train()` / `eval()` modes (similar to PyTorch)  
* Zeroing gradients via `zero_grad()`

#### **Neuron**

Represents a single fully connected neuron:

```python
from autograd.nn import Neuron
n = Neuron(nin=3, act_func="relu")
```

Each neuron maintains its own weights, bias, and activation function.

#### **Layer**

Wraps multiple neurons into a dense layer:

```python
layer = Layer(nin=3, nout=4, act_func="tanh")
out = layer([Value(1), Value(2), Value(3)])
```

#### **MLP (Multi-Layer Perceptron)**

Stacks layers sequentially to form a deep neural network:

```python
from autograd.nn import MLP
model = MLP(nin=3, nouts=[4, 4, 1], act_func="relu")
```

##### Additional Features

* `summary()` prints a readable model overview  
* Adaptive learning rate logic inside `update(loss)`

---

### **3. Graph Visualization (graph.py)**

Uses **Graphviz** to render the computational graph created during forward and backward passes.

#### Example

```python
from autograd.graph import draw_graph
from autograd.engine import Value

x = Value(2.0, label='x')
y = Value(3.0, label='y')
z = x * y + x.relu()
z.backward()

dot = draw_graph(z)
dot.render("graph", view=True)
```

Output: A clean, left-to-right computational graph showing nodes, operations, data, and gradients.

---

## 🧪 Example: Training a Tiny Neural Network

```python
from autograd.nn import MLP
from autograd.engine import Value

# Dummy dataset
X = [
  [Value(2.0), Value(3.0), Value(-1.0)],
  [Value(3.0), Value(-1.0), Value(0.5)],
]
Y = [Value(1.0), Value(-1.0)]

model = MLP(nin=3, nouts=[4, 4, 1], act_func="tanh")

for epoch in range(100):
    total_loss = Value(0.0)
    for x, y in zip(X, Y):
        pred = model(x)
        loss = (pred - y)**2
        total_loss += loss

    model.zero_grad()
    total_loss.backward()
    model.update(total_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss: {total_loss.data:.4f} | LR: {model.lr:.5f}")
```

---

## 📈 Highlights

✅ Fully vector-free scalar autograd (manual computation graph)  
✅ Dynamic LR scheduling built into the training loop  
✅ Clean modular OOP design (PyTorch-like hierarchy)  
✅ Support for advanced activations and operations  
✅ Visualization with Graphviz  
✅ Compact yet expressive educational codebase

---

## 🧰 Requirements

* Python ≥ 3.8  
* `graphviz`  
* `torch` (optional, used only for safe tensor casting)

Install dependencies:

```bash
pip install graphviz torch
```

---

## 🧩 Future Improvements

* Add mini-batch data loaders  
* Integrate support for vectorized `Value` arrays  
* Implement optimizers (SGD, Adam)  
* Extend visualization with color-coded gradients

---

## 🧑‍💻 Author

**Adnan Alaref**  
Machine Learning Engineer & 2x Kaggle Expert  

> *“Understand backpropagation by building it from scratch.”*

---

## 🧾 License
**MIT License** © 2025 Adnan Alaref  
You are free to use, modify, and distribute it with attribution.
