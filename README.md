# Neural Networks — Homework

Homework solutions for the **Neural Networks** course, Artificial Intelligence program, Alexandru Ioan Cuza University of Iași.

**Author:** Elisei Chiriac

## About

This repository collects the assignments for the course, one folder per homework. Each assignment builds on the previous ones, starting from the linear algebra and Python foundations needed to implement neural networks.

## Requirements

- Python 3.10+
- Unless an assignment says otherwise, no external libraries (no NumPy or other linear algebra frameworks); matrices and vectors are plain Python lists.

## Repository structure

```
NN_homework/
├── README.md
├── 1st_homework/
│   ├── ex1/
│   │   ├── ex1.py
│   │   └── ex1.txt
│   └── ex2/
│       └── ex2.py
└── ...
```

## Homework overview

| # | Topic | Folder | Status |
|---|---|---|---|
| 1 | Solving a linear system in Python | `1st_homework/` | ✅ Done |

---

## Homework 1 — Solving a linear system in Python (10 points)

Solve a system of 3 linear equations with 3 unknowns, written in vector form as **A·X = B**, using only Python lists.

### Tasks

| Task | Points | Description |
|---|---|---|
| 1. Parsing | 1 | Read the system from a text file into matrix `A` and vector `B` |
| 2. Matrix and vector operations | 5 | Determinant, trace, Euclidean norm, transpose, matrix–vector multiplication |
| 3. Cramer's rule | 1 | Solve by replacing each column of `A` with `B` |
| 4. Inversion | 3 | Solve with `A⁻¹ = adj(A) / det(A)`, using minors and cofactors |

### Input format

Coefficients are always in the order `x`, `y`, `z`, and terms are space separated:

```
2x + 3y - z = 5
x - y + 4z = 6
3x + y + 2z = 7
```

### How to run

Parse the system (writes `ex1.out`):

```
cd 1st_homework/ex1
python ex1.py
```

Run the operations and both solvers (from `1st_homework`, so that `ex2` can import from `ex1`):

```
cd 1st_homework
python -m ex2.ex2
```

### Example

For the system above:

| Result | Value |
|---|---|
| det(A) | 14 |
| tr(A) | 3 |
| ‖B‖ | √110 ≈ 10.488 |
| Solution (Cramer and inversion) | x = 5/14 ≈ 0.357, y = 29/14 ≈ 2.071, z = 27/14 ≈ 1.929 |

### Notes

- If `det(A)` is (close to) 0, the system has no unique solution and both solvers raise an error.
- Any solution can be checked by multiplying `A` with it: the result must be `B`.

---

<!--
Template for the next homework — copy, fill in, and add a row to the overview table.

## Homework N — Title (X points)

Short description of the goal.

### Tasks

| Task | Points | Description |
|---|---|---|
| 1. ... | | |

### How to run

```
cd Nth_homework
python ...
```

### Results / notes

-->
# NN_homework