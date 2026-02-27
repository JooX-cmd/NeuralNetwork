# Homework 2 - Backpropagation

---

## Problem #1: Sklearn - Neural Network Regression

### Goal
Find the best Neural Network architecture that gives
the lowest validation RMSE on `data2_200x30.csv`

### Dataset
- 200 samples × 30 features
- Train/Val split: 100/100
- Missing values filled with median

### Approach
Experimented with different combinations of:

| Parameter | Options Tried |
|---|---|
| Architecture | (5,5,5), (100,), (50,25), (10,10,10), (64,32,16) |
| Activation | relu, tanh |
| Solver | adam, sgd |
| Preprocessing | MinMax(1), Standardize(2) |

### Results

| Model | Architecture | Activation | Solver | RMSE |
|---|---|---|---|---|
| Baseline | (5, 5, 5) | relu | adam | 12.6358 |
| **Best** | **(64, 32, 16)** | **tanh** | **adam** | **12.3632** |

### Why (64, 32, 16) + tanh + adam Won?
- **Pyramid shape** → learns broad features first, refines them later
- **tanh** → better than relu for this regression problem
- **adam** → adaptive optimizer, adjusts learning rate automatically
- **sgd** performed very poorly (RMSE > 70) showing adam is critical here

### How to Run
```bash
# Baseline
python 1_skit.py --preprocessing 1

# With standardization
python 1_skit.py --preprocessing 2
```

### Requirements
```
numpy
pandas
scikit-learn
```

---

## Problem #2: Backpropagation from Scratch

### Goal
Implement a Neural Network from scratch using only Python + NumPy.  
No sklearn. Just pure math and code.

### Network Architecture
```
Input Layer → Hidden Layer → Output Layer
x1, x2     →  h1, h2      →  o1, o2
```

### Classes Built

| Class | Responsibility |
|---|---|
| `Neuron` | Single neuron: computes net, activation, derivative |
| `NeuronLayer` | Group of neurons: passes input through all neurons |
| `NeuralNetwork` | Full network: feed forward + backprop + weight update |

### The 3 Steps Per Training Iteration

```
Step 1: FEED FORWARD
        net    = Σ(weight × input)   ← dot product
        output = activation(net)      ← apply f(net)

Step 2: COMPUTE DELTA (Backpropagation)
        Output layer: δ_o = (output - target) × f'(net)
        Hidden layer: δ_h = (Σ δ_o × w_ho) × f'(net)

Step 3: UPDATE WEIGHTS
        new_w = old_w - lr × δ × input
```

### Activation Functions Supported

| Function | Formula | Derivative |
|---|---|---|
| Polynomial | f(net) = net² | f'(net) = 2 × net |
| Sigmoid | f(net) = 1/(1+e^-net) | f'(net) = out × (1-out) |
| Identity | f(net) = net | f'(net) = 1 |

### Test 1: Polynomial (2×2×2 Network)

**Input:** `[1, 1]` → **Target:** `[290, 14]`

```
network output: [289, 16]
Delta o[0]: -34.0
Delta o[1]: 16.0
Delta h[0]: -208.0
Delta h[1]: -204.0
node o: 0 - w_ho: 0: Delata -136.0 => new w = 70.0
node o: 0 - w_ho: 1: Delata -306.0 => new w = 154.0
node o: 1 - w_ho: 0: Delata 64.0   => new w = -31.0
node o: 1 - w_ho: 1: Delata 144.0  => new w = -72.0
node h: 0 - w_ih: 0: Delata -208.0 => new w = 105.0
node h: 0 - w_ih: 1: Delata -208.0 => new w = 105.0
node h: 1 - w_ih: 0: Delata -204.0 => new w = 104.0
node h: 1 - w_ih: 1: Delata -204.0 => new w = 103.0
```
✅ All values match expected output

### Test 2: Sigmoid (2×4×3 Network)

**Input:** `[1, 2]` → **Target:** `[0.4, 0.7, 0.6]`

```
network output: [0.5913, 0.6219, 0.6508]
Delta o[0]:  0.04623
Delta o[1]: -0.01835
Delta o[2]:  0.01155
Delta h[0]:  0.000963
Delta h[1]:  0.002891
Delta h[2]:  0.001386
Delta h[3]:  0.000556
```
✅ All values match expected output

### How to Run

```bash
# Run polynomial test
python 2_homework_backpropagation_template.py

# To run sigmoid test → change last line in file:
# poly()  ← comment this
# sigm()  ← uncomment this
```

### Key Bug Fixes During Implementation

| Bug | Wrong | Fixed |
|---|---|---|
| Wrong variable name | `calc_net_out(input)` | `calc_net_out(inputs)` |
| Print outside loop | `print(Delta H: array)` | `print(f'Delta h[{h}]...')` inside loop |

### Requirements
```
numpy
math
```

---

## File Structure

```
📁 7 Homework 2 - Backpropagation
    ├── 📄 README.md                              ← this file
    ├── 🐍 1_skit.py                              ← Problem 1: Sklearn
    ├── 🐍 2_homework_backpropagation_template.py ← Problem 2: From scratch
    ├── 🐍 data_helper.py                         ← Data loading & preprocessing
    └── 📊 data2_200x30.csv                       ← Dataset
```