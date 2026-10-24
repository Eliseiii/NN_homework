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
