import math

def determinant(A):
    return float(A[0][0]* (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                 +A[0][1]* (A[1][2] * A[2][0] - A[1][0] * A[2][2])
                 +A[0][2]* (A[1][0] * A[2][1] - A[1][1] * A[2][0]))

def trace(A):
    return float(A[0][0]+A[1][1]+A[2][2])

def norm(b):
    return math.sqrt(sum([x**2 for x in b]))

def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]

def  mat_vec_mult(A,b):
    return [sum(A[i][j] * b[j] for j in range(3)) for i in range(3)]

def solve_cramer(A,b):
    eps=1e-6
    det_A = determinant(A)
    if abs(det_A) < eps:
        raise ValueError("The system has no unique solution.")
    
    x = [0, 0, 0]
    for i in range(3):
        A_i = [row[:] for row in A]
        for r in range(3):
            A_i[r][i] = b[r]
        x[i] = determinant(A_i) / det_A
    
    return x

def det2(A):
    return A[0][0]*A[1][1]-A[0][1]*A[1][0]

def solve(A, b):
    adjA = [[0, 0, 0] for _ in range(3)]
    for i in range(3):  
        for j in range(3):
            minor = [[A[m][n] for n in range(3) if n != j] for m in range(3) if m != i]
            adjA[j][i] = ((-1) ** (i + j)) * det2(minor)

    detA = determinant(A)
    if abs(detA) < 1e-6:
        raise ValueError("The system has no unique solution.")
    return [x / detA for x in mat_vec_mult(adjA, b)]


def main():
    A=[[2.0, 3.0, -1.0], [1.0, -1.0, 4.0], [3.0, 1.0, 2.0]]
    b=[1.0, 2.0, 3.0]
    print("Matrix A:")
    print(A)
    print("Vector b:")
    print(b)

    det_A = determinant(A)
    tr_A = trace(A)
    norm_b = norm(b)
    trans_A = transpose(A)

    print(f"Determinant of A: {det_A}")
    print(f"Trace of A: {tr_A}")
    print(f"Norm of b: {norm_b}")
    print("Transpose of A:")
    print(trans_A)

    try:
        solution_cramer = solve_cramer(A, b)
        print(f"Solution using Cramer's rule: {solution_cramer}")
    except ValueError as e:
        print(e)

    try:
        solution = solve(A, b)
        print(f"Solution using adjugate method: {solution}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()