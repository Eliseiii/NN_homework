
def perse_exuasion(line, unknows="xyz"):
    left, right=line.replace(" ","").split("=")
    coef=[0.0]*3
    sign=1
    num=""
    for c in left:
        if c=="-":
            sign=-1
        elif c== "+":
            sign=1
        elif c in unknows:
            value=float(num) if num else 1.0
            coef[unknows.index(c)]+=sign*value
            sign=1
            num=""
        else:
            num+=c
    return coef, float(right)

A=[]
b=[]

def main():
    with open("ex1.txt") as f:
     for line in f:
        coef, value=perse_exuasion(line)
        A.append(coef)
        b.append(value)

    with open("ex1.out", "w") as f:
        print(A, file=f)
        print(b, file=f)

    def get_mat_vec():
        return A, b

if __name__ == "__main__":
    main()
