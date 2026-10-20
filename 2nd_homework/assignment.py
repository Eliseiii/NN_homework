import numpy as np

kaggle_data="/kaggle/input/competitions/nn-2026-fii-homework-1"

def load_data():
    try:
        train = np.load(f"{kaggle_data}/train.npz")
        test = np.load(f"{kaggle_data}/test.npz")
    except FileNotFoundError:
        print("Data files not found.")
        return None, None, None

    X_train = train["data"].astype(np.float32)
    y_train = train["labels"].astype(np.int32)
    X_test = test["data"].astype(np.float32)

    if X_train.max() > 1:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    return X_train, y_train, X_test


def softmax(x):
    x=x-x.max(axis=1, keepdims=True) #posibility of overflow when calculating exp(x)
    exp_x=np.exp(x)
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def forward_prop(X,W,b):
    return softmax(X@W+b)

def cross_entropy(y_true, y_pred):
    eps=1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred+eps), axis=1))

def backward(X, y_pred, y_true, W, b, lr):
    m=X.shape[0]
    error=y_true-y_pred
    W=W+ lr*(X.T@error)/m
    b=b+ lr*error.sum(axis=0)/m
    return W, b

def accuracy(X, labels, W, b):
    preds= np.argmax(forward_prop(X, W, b), axis=1)
    return np.mean(preds==labels)

def train(x, labels, epochs=40, lr=0.5, batch_size=100):
    T=np.eye(10, dtype=np.float32)[labels]
    W=np.zeros((784,10), dtype=np.float32)
    b=np.zeros((10), dtype=np.float32)
    

    for epoch in range(epochs):
        cr_lr=float(lr*0.5*(1+np.cos(np.pi*epoch/epochs))) #cosine annealing
        order=np.random.permutation(len(x))
        
        for start in range(0, len(x), batch_size):
           idx=order[start:start+batch_size]
           y_pred=forward_prop(x[idx], W, b)
           W, b=backward(x[idx], y_pred, T[idx], W, b, cr_lr)

        if (epoch+1) % 5 == 0:
            loss=cross_entropy(T, forward_prop(x, W, b))
            acc=accuracy(x, labels, W, b)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    return W, b

def predict(X,W,b):
    return np.argmax(X@W+b, axis=1)

def split(X, labels,val_ratio=0.2):
    order=np.random.permutation(len(X))
    n_val=int(len(X)*val_ratio)
    val_idx, train_idx=order[:n_val], order[n_val:]
    return X[train_idx], labels[train_idx], X[val_idx], labels[val_idx]

def save_csv(predictions, filename="submission.csv"):
    with open(filename, "w") as f:
        f.write("ID,target\n")
        for i, p in enumerate(predictions):
            f.write(f"{i},{p}\n")

    print("created submission.csv file")


def main():
    np.random.seed(1)

    train_X, train_y, test_X = load_data()

    X_train, y_train, X_val, y_val = split(train_X, train_y)
    W, b = train(X_train, y_train, epochs=40, lr=0.5, batch_size=100)
    print("accuracy:", accuracy(X_val, y_val, W, b))

    W, b = train(train_X, train_y, epochs=40, lr=0.5, batch_size=100)
    save_csv(predictions=np.argmax(forward_prop(test_X, W, b), axis=1))

if __name__ == "__main__":
    main()
