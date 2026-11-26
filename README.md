# Neural Networks from Scratch

Three small projects that build up a neural network from the ground up: first the linear algebra it rests on, then a single-layer classifier, then a full multi-layer network with backpropagation. Everything is written by hand in Python, without deep learning frameworks.

| # | Project | What it does | Stack |
|---|---------|--------------|-------|
| 1 | [Linear System Solver](#1-linear-system-solver) | Parses and solves a 3×3 system of equations | Python (standard library) |
| 2 | [Softmax Perceptron](#2-softmax-perceptron) | Single-layer classifier for 28×28 images | Python, NumPy |
| 3 | [MLP with Backpropagation](#3-mlp-with-backpropagation) | Two-layer network with an optimized training pipeline | Python, NumPy, SciPy |

## Repository structure

```
.
├── 1st_homework/
│   ├── Assignment1.ipynb        # problem description and function skeletons
│   ├── ex1/                     # equation parser (ex1.py, ex1.txt, ex1.out)
│   └── ex2/ex2.py               # matrix operations, Cramer's rule, adjugate inversion
├── 2nd_homework/
│   ├── assignment.py            # softmax perceptron
│   └── assignment2.pdf          # problem description
├── 3rd_homework/
│   └── mlp.py                   # MLP with backpropagation
└── README.md
```

---

## 1. Linear System Solver

Solves a system of three linear equations `A·x = b` using only Python lists. No NumPy, so every step of the linear algebra is explicit.

**What it does**
- **Parser** (`ex1/ex1.py`): reads equations such as `2x + 3y - z = 5` from a text file and builds the coefficient matrix `A` and the vector `b`.
- **Matrix and vector operations** (`ex2/ex2.py`): determinant, trace, vector norm, transpose and matrix-vector product.
- **Two solvers**: Cramer's rule, and inversion through the adjugate matrix (minors → cofactors → adjugate → `A⁻¹·b`). Both raise an error when the determinant is close to zero, i.e. when there is no unique solution.

```bash
cd 1st_homework/ex1 && python ex1.py     # parse ex1.txt, write ex1.out
cd ../ex2 && python ex2.py               # run the operations and both solvers
```

For the sample system in `ex2.py`, both solvers return the same solution up to floating-point rounding.

---

## 2. Softmax Perceptron

A single-layer classifier, equivalent to ten perceptrons (one per class), trained on flattened 28×28 images (784 features, 10 classes).

**What it does**
- **Forward pass**: `softmax(X @ W + b)`, with the maximum subtracted before exponentiation for numerical stability.
- **Loss and update**: cross-entropy loss and the gradient-descent update `W ← W + lr · Xᵀ(target − y) / m`, with the same rule for the bias.
- **Training**: mini-batches of 100 with reshuffling every epoch, 40 epochs, and a learning rate of 0.5 decayed with cosine annealing.
- **Evaluation**: an 80/20 split to measure validation accuracy, then a retrain on the full training set to produce `submission.csv` (`ID,target`).

```bash
pip install numpy
python 2nd_homework/assignment.py
```

The script reads `train.npz` and `test.npz` from the Kaggle input folder; change `kaggle_data` at the top of the file to run it elsewhere.

---

## 3. MLP with Backpropagation

A fully-connected network (784 → 100 → 10) with hand-written backpropagation, designed to reach high accuracy on a plain two-layer architecture within a fixed time budget.

**Model**
- Sigmoid hidden layer, softmax output, cross-entropy loss, Glorot-normal initialization.
- Analytic gradients, fully batched: `δ₂ = a₂ − t`, `δ₁ = (W₂ᵀ δ₂) ⊙ a₁ ⊙ (1 − a₁)`.
- Nesterov momentum (0.9), L2 weight decay (1e-4) and cosine learning-rate annealing from 0.3.

**Data pipeline**
- **Pixel-order recovery**: the feature columns are not stored in image order, so the 28×28 layout is reconstructed from pixel correlations (nearest-neighbour graph, geodesic distances, classical MDS, Hungarian assignment and local swap refinement).
- **Deskewing**: moment-based shear correction with bilinear interpolation.
- **Augmentation**: random rotation (±8°), scale (±8 %) and shift (±1.5 px), applied on the fly to each mini-batch.

**Training**
- 5,000 samples held out for validation, mini-batches of 128, up to 160 epochs.
- The learning-rate schedule follows the wall clock as well as the step count, so training always finishes within the 330 s budget.
- Per-epoch training and validation loss and accuracy are printed; predictions are saved to `submission.csv`.

```bash
pip install numpy scipy
python 3rd_homework/mlp.py
```

Data is looked up under `/kaggle/input/**` and otherwise in the current directory. Hyperparameters are constants at the top of the script.

---

## Requirements

- Python 3.9+
- Project 1: standard library only
- Project 2: `numpy`
- Project 3: `numpy`, `scipy`

