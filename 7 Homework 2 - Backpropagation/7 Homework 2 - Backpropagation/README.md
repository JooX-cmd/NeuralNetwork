# Homework 2 - Backpropagation

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