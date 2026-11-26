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

# Neural Networks – Homework

Solutions for the homework assignments of the **Neural Networks** course (FII, UAIC, 2026).

| Assignment | Topic | Folder |
|------------|-------|--------|
| 1 | Solving a 3x3 linear system in pure Python | [`1st_homework/`](1st_homework) |
| 2 | Perceptron (softmax classifier, 784 inputs / 10 outputs) in NumPy | [`2nd_homework/`](2nd_homework) |

## Repository structure

```
NN_homework/
├── 1st_homework/
│   ├── Assignment1.ipynb      # assignment statement + function skeletons
│   ├── ex1/
│   │   ├── ex1.py             # parser for the system of equations
│   │   ├── ex1.txt            # input: the system of equations
│   │   └── ex1.out            # output: matrix A and vector b
│   └── ex2/
│       └── ex2.py             # matrix/vector operations + Cramer + adjugate inversion
├── 2nd_homework/
│   ├── assignment.py          # perceptron: training + Kaggle submission
│   └── assignment2.pdf        # assignment statement
└── README.md
```

---

## Assignment 1 – Solving a linear system in Python

**Statement** (see `1st_homework/Assignment1.ipynb`): solve the system

```
2x + 3y - z = 5
x - y + 4z = 6
3x + y + 2z = 7
```

written as `A · X = B`, using only Python lists (no NumPy or other linear algebra framework).

### 1. Parsing the system (`ex1/ex1.py`)

Reads the system from `ex1.txt` and builds the matrix `A` and the vector `b`.

- Each line is stripped of spaces and split on `=`.
- The left side is scanned character by character: the sign (`+`/`-`) and the numeric coefficient are accumulated until a variable (`x`, `y`, `z`) is reached; a missing coefficient means `1`.
- Coefficients are stored in the order x, y, z.
- The result is written to `ex1.out`.

Output for the given system:

```
[[2.0, 3.0, -1.0], [1.0, -1.0, 4.0], [3.0, 1.0, 2.0]]
[5.0, 6.0, 7.0]
```

Run (from the `ex1` folder, since the file is opened by relative path):

```bash
cd 1st_homework/ex1
python ex1.py
```

### 2. Matrix and vector operations (`ex2/ex2.py`)

| Function | Description |
|----------|-------------|
| `determinant(A)` | Determinant of a 3x3 matrix, by cofactor expansion along the first row |
| `trace(A)` | Sum of the main diagonal |
| `norm(b)` | Euclidean norm of a vector |
| `transpose(A)` | Transpose of a 3x3 matrix |
| `mat_vec_mult(A, b)` | Matrix-vector product |

### 3. Cramer's rule (`ex2/ex2.py`)

`solve_cramer(A, b)` computes `det(A)`, then for each unknown replaces the corresponding column of `A` with `b` and divides the new determinant by `det(A)`. If `|det(A)| < 1e-6` the system has no unique solution and a `ValueError` is raised.

### 4. Solving by inversion (`ex2/ex2.py`)

`solve(A, b)` uses the adjugate method, `A⁻¹ = adj(A) / det(A)`:

- the minor of each element is obtained by removing its row and column (`det2` computes the 2x2 determinant);
- the cofactor is `(-1)^(i+j) · det(minor)`;
- the adjugate is the transpose of the cofactor matrix;
- the solution is `X = A⁻¹ · b`.

The same singularity check as in Cramer's rule is applied.

Run:

```bash
cd 1st_homework/ex2
python ex2.py
```

`main()` uses a different set of coefficients from the statement (`A[0][0] = 0.1`, `b = [1, 2, 3]`), as the assignment suggests experimenting with other values. Output:

```
Matrix A:
[[0.1, 3.0, -1.0], [1.0, -1.0, 4.0], [3.0, 1.0, 2.0]]
Vector b:
[1.0, 2.0, 3.0]
Determinant of A: 25.4
Trace of A: 1.1
Norm of b: 3.7416573867739413
Transpose of A:
[[0.1, 1.0, 3.0], [3.0, -1.0, 1.0], [-1.0, 4.0, 2.0]]
Solution using Cramer's rule: [0.5118110236220472, 0.48031496062992124, 0.4921259842519685]
Solution using adjugate method: [0.5118110236220472, 0.48031496062992135, 0.4921259842519685]
```

Both methods agree up to floating-point rounding.

---

## Assignment 2 – Perceptron

**Statement** (see `2nd_homework/assignment2.pdf`): implement forward and backward propagation, using NumPy, for a network with 784 inputs and 10 outputs, which can be seen as 10 perceptrons, one per class. The submission is made through a Kaggle competition.

### Implementation (`2nd_homework/assignment.py`)

**Data.** `load_data()` reads `train.npz` and `test.npz` from the Kaggle input folder (`data` and `labels` arrays). Pixels are scaled to `[0, 1]` when the maximum value is greater than 1.

**Forward propagation.** `forward_prop(X, W, b) = softmax(X @ W + b)`, with `X` of shape `(m, 784)`, `W` of shape `(784, 10)` and `b` of shape `(10,)`. The softmax subtracts the row maximum before exponentiation to avoid overflow.

**Loss.** `cross_entropy(y_true, y_pred)`, averaged over the batch, with a small `eps` inside the logarithm for numerical stability.

**Backward propagation / update.** Gradient descent on the cross-entropy loss, as in equations (2) and (3) of the statement:

```
W = W + lr · Xᵀ · (Target − y) / m
b = b + lr · sum(Target − y) / m
```

where `Target` is the one-hot encoding of the labels.

**Training (`train`).**

- weights initialized with zeros, mini-batch gradient descent with shuffling at every epoch;
- default hyperparameters: 40 epochs, learning rate 0.5, batch size 100;
- learning rate decayed with **cosine annealing**: `lr_t = lr · 0.5 · (1 + cos(π · t / epochs))`;
- loss and accuracy are printed every 5 epochs.

**Evaluation and submission (`main`).**

1. fixed random seed (`np.random.seed(1)`);
2. the training set is split 80% / 20% (`split`) and the model is trained on the 80% part to measure the validation accuracy;
3. the model is retrained on the **full** training set and the predictions on the test set (`argmax` of the softmax output) are saved to `submission.csv` (`ID,target`).

### Running

The script expects the competition data at `/kaggle/input/competitions/nn-2026-fii-homework-1` (the path of the Kaggle environment, set by the `kaggle_data` variable). To run it locally, change that variable to the folder containing `train.npz` and `test.npz`:

```bash
pip install numpy
python 2nd_homework/assignment.py
```

It prints the training progress, the validation accuracy and creates `submission.csv` in the current folder.

---

## Requirements

- Python 3.10+
- Assignment 1: standard library only
- Assignment 2: `numpy`
