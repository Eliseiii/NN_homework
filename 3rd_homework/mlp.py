import glob
import os
import time
 
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import shortest_path
 
SEED = 0
LAYER_SIZES = (784, 100, 10)
 
EPOCHS = 160
BATCH_SIZE = 128
LEARNING_RATE = 0.3
MOMENTUM = 0.9
L2_LAMBDA = 1e-4
VAL_SIZE = 5000
AUGMENT = True
TOTAL_BUDGET = 330
 
PIXEL_MEAN, PIXEL_STD = 0.0, 1.0
 
 
def find_data():
    folder = "."
    for train_file in glob.glob("/kaggle/input/**/train.npz", recursive=True):
        if os.path.exists(os.path.join(os.path.dirname(train_file), "test.npz")):
            folder = os.path.dirname(train_file)
            break
 
    train = np.load(os.path.join(folder, "train.npz"))
    test = np.load(os.path.join(folder, "test.npz"))
 
    X_train = train["data"].reshape(len(train["data"]), -1).astype(np.float32)
    y_train = train["labels"].astype(np.int64)
    X_test = test["data"].reshape(len(test["data"]), -1).astype(np.float32)
 
    if X_train.max() > 1:
        X_train /= 255.0
        X_test /= 255.0
 
    return X_train, y_train, X_test
 
 
def recover_pixel_order(X, k=6):
    Xc = X - X.mean(0)
    std = Xc.std(0)
    active = std > 1e-6
    C = np.zeros((784, 784))
    C[np.ix_(active, active)] = np.corrcoef(Xc[:, active].T)
    np.fill_diagonal(C, 0)
 
    ia = np.where(active)[0]
    n = len(ia)
    Ca = C[np.ix_(ia, ia)]
    nn = np.argsort(-Ca, axis=1)[:, :k]
    G = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    G[rows, nn.ravel()] = -np.log(np.clip(Ca[rows, nn.ravel()], 1e-6, 0.999999))
    G = np.maximum(G, G.T)
    D = shortest_path(G, directed=False)
    D[np.isinf(D)] = D[~np.isinf(D)].max()
    J = np.eye(n) - 1 / n
    w, v = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    emb = v[:, -2:] * np.sqrt(np.maximum(w[-2:], 0))
    emb -= emb.mean(0)
 
    cells = np.array([(r, c) for r in range(28) for c in range(28)], dtype=float)
 
    def assign(e):
        cost = ((e[:, None, :] - cells[None]) ** 2).sum(-1)
        ri, ci = linear_sum_assignment(cost)
        pos = np.full(784, -1)
        pos[ia[ri]] = (cells[ci, 0] * 28 + cells[ci, 1]).astype(int)
        free = np.setdiff1d(np.arange(784), pos[ia])
        pos[~active] = free
        return pos
 
    def objective(pos):
        g = np.empty(784, dtype=int)
        g[pos] = np.arange(784)
        g = g.reshape(28, 28)
        return C[g[:, :-1], g[:, 1:]].sum() + C[g[:-1], g[1:]].sum()
 
    best_score, best_pos = -np.inf, None
    for sc in (0.85, 0.95, 1.05):
        for ang in np.deg2rad(np.arange(0, 90, 5)):
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            e = emb @ R
            e = e / e.std() * 7.6 * sc + 13.5
            pos = assign(e)
            score = objective(pos)
            if score > best_score:
                best_score, best_pos = score, pos
 
    pos_r, pos_c = best_pos // 28, best_pos % 28
    grid = np.full((32, 32), 784)
    grid[pos_r + 2, pos_c + 2] = np.arange(784)
    Cp = np.zeros((785, 785))
    Cp[:784, :784] = C
    offs = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)])
    wts = np.array([0.5, 1, 0.5, 1, 1, 0.5, 1, 0.5])
    top = np.argsort(-C, axis=1)[:, :6]
 
    def local(j, r, c, other):
        nb = grid[r + 2 + offs[:, 0], c + 2 + offs[:, 1]]
        nb = np.where(nb == other, 784, nb)
        return (wts * Cp[j, nb]).sum()
 
    for _ in range(15):
        swaps = 0
        for a in range(784):
            cands = set()
            for b in top[a]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r2, c2 = pos_r[b] + dr, pos_c[b] + dc
                        if 0 <= r2 < 28 and 0 <= c2 < 28:
                            cands.add(int(grid[r2 + 2, c2 + 2]))
            for b in cands:
                if b == a:
                    continue
                ra, ca, rb, cb = pos_r[a], pos_c[a], pos_r[b], pos_c[b]
                old = local(a, ra, ca, b) + local(b, rb, cb, a)
                new = local(a, rb, cb, b) + local(b, ra, ca, a)
                if new > old + 1e-9:
                    grid[ra + 2, ca + 2], grid[rb + 2, cb + 2] = b, a
                    pos_r[a], pos_c[a], pos_r[b], pos_c[b] = rb, cb, ra, ca
                    swaps += 1
        if swaps == 0:
            break
 
    order = np.argsort(pos_r * 28 + pos_c)
 
    im = X[:, order].reshape(-1, 28, 28)
    rr, cc = np.mgrid[0:28, 0:28]
    tot = im.sum((1, 2)) + 1e-8
    mr = (im * rr).sum((1, 2)) / tot
    mc = (im * cc).sum((1, 2)) / tot
    spread_r = ((im * (rr - mr[:, None, None]) ** 2).sum((1, 2)) / tot).mean()
    spread_c = ((im * (cc - mc[:, None, None]) ** 2).sum((1, 2)) / tot).mean()
    if spread_c > spread_r:
        order = order.reshape(28, 28).T.ravel()
    return order
 
 
def deskew(X, chunk=10000):
    out = np.empty_like(X, dtype=np.float32)
    rr, cc = np.mgrid[0:28, 0:28].astype(np.float32)
    for s in range(0, len(X), chunk):
        im = X[s:s + chunk].reshape(-1, 28, 28).astype(np.float32)
        n = len(im)
        tot = im.sum((1, 2)) + 1e-8
        mr = (im * rr).sum((1, 2)) / tot
        mc = (im * cc).sum((1, 2)) / tot
        vr = (im * (rr - mr[:, None, None]) ** 2).sum((1, 2)) / tot
        cov = (im * (rr - mr[:, None, None]) * (cc - mc[:, None, None])).sum((1, 2)) / tot
        alpha = cov / (vr + 1e-8)
        src_r = rr[None] + (mr - 14.0)[:, None, None]
        src_c = cc[None] + alpha[:, None, None] * (rr[None] - 14.0) + (mc - 14.0)[:, None, None]
        r0, c0 = np.floor(src_r).astype(np.int32), np.floor(src_c).astype(np.int32)
        fr, fc = src_r - r0, src_c - c0
        padded = np.pad(im, ((0, 0), (1, 1), (1, 1)))
        k = np.arange(n)[:, None, None]
 
        def px(r, c):
            return padded[k, np.clip(r + 1, 0, 29), np.clip(c + 1, 0, 29)]
 
        out[s:s + chunk] = ((1 - fr) * (1 - fc) * px(r0, c0) + (1 - fr) * fc * px(r0, c0 + 1)
                            + fr * (1 - fc) * px(r0 + 1, c0) + fr * fc * px(r0 + 1, c0 + 1)).reshape(n, 784)
    return out
 
 
def split_data(X, y, val_size, rng):
    order = rng.permutation(len(X))
    val_idx = order[:val_size]
    train_idx = order[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
 
 
_ROWS, _COLS = np.meshgrid(np.arange(28) - 13.5, np.arange(28) - 13.5, indexing="ij")
_GRID = np.stack([_ROWS.ravel(), _COLS.ravel(), np.ones(784)]).astype(np.float32)
 
 
def augment(x, rng, max_rotation=8.0, max_scale=0.08, max_shift=1.5):
    m = len(x)
    angle = np.deg2rad(rng.uniform(-max_rotation, max_rotation, m))
    scale = rng.uniform(1 - max_scale, 1 + max_scale, m)
    shift = rng.uniform(-max_shift, max_shift, (m, 2)) + 14.5
    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
 
    A = np.empty((m, 2, 3), dtype=np.float32)
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = cos, -sin, shift[:, 0]
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = sin, cos, shift[:, 1]
 
    src = A @ _GRID
    np.clip(src, 0, 28.999, out=src)
 
    top_left = src.astype(np.int32)
    w_row, w_col = src[:, 0] - top_left[:, 0], src[:, 1] - top_left[:, 1]
 
    padded = np.zeros((m, 30, 30), dtype=np.float32)
    padded[:, 1:29, 1:29] = x.reshape(m, 28, 28)
    flat = padded.reshape(-1)
    idx = top_left[:, 0] * 30 + top_left[:, 1] + (np.arange(m, dtype=np.int32) * 900)[:, None]
    p00, p01 = np.take(flat, idx), np.take(flat, idx + 1)
    p10, p11 = np.take(flat, idx + 30), np.take(flat, idx + 31)
 
    top = p00 + (p01 - p00) * w_col
    bottom = p10 + (p11 - p10) * w_col
    return top + (bottom - top) * w_row
 
 
def set_normalization(X_train):
    global PIXEL_MEAN, PIXEL_STD
    PIXEL_MEAN, PIXEL_STD = float(X_train.mean()), float(X_train.std())
 
 
def normalize(x):
    return (x - PIXEL_MEAN) / PIXEL_STD
 
 
def softmax(z):
    z = z - z.max(axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=0, keepdims=True)
 
 
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))
 
 
def cross_entropy(probs, labels):
    return float(-np.mean(np.log(probs[labels, np.arange(len(labels))] + 1e-12)))
 
 
def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes, dtype=np.float32)[y].T
 
 
class MLP:
    def __init__(self, sizes=LAYER_SIZES, seed=SEED, dtype=np.float32):
        n_in, n_hidden, n_out = sizes
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2 / (n_in + n_hidden)), (n_hidden, n_in)).astype(dtype)
        self.b1 = np.zeros((n_hidden, 1), dtype=dtype)
        self.W2 = rng.normal(0, np.sqrt(2 / (n_hidden + n_out)), (n_out, n_hidden)).astype(dtype)
        self.b2 = np.zeros((n_out, 1), dtype=dtype)
        self.momentum_buffers = [np.zeros_like(p) for p in self.parameters()]
 
    def parameters(self):
        return [self.W1, self.b1, self.W2, self.b2]
 
    def forward(self, X):
        a0 = X.T
        z1 = self.W1 @ a0 + self.b1
        a1 = sigmoid(z1)
        z2 = self.W2 @ a1 + self.b2
        a2 = softmax(z2)
        return a2, (a0, a1)
 
    def backward(self, a2, t, cache):
        a0, a1 = cache
        m = t.shape[1]
 
        delta2 = a2 - t
        delta1 = (self.W2.T @ delta2) * a1 * (1 - a1)
 
        dW2 = delta2 @ a1.T / m
        db2 = delta2.sum(axis=1, keepdims=True) / m
        dW1 = delta1 @ a0.T / m
        db1 = delta1.sum(axis=1, keepdims=True) / m
        return [dW1, db1, dW2, db2]
 
    def update(self, grads, lr, momentum=MOMENTUM, l2_lambda=L2_LAMBDA):
        is_weight = (True, False, True, False)
        for param, grad, buf, weight in zip(self.parameters(), grads, self.momentum_buffers, is_weight):
            if weight and l2_lambda:
                grad = grad + l2_lambda * param
            buf *= momentum
            buf += grad
            param -= lr * (grad + momentum * buf)
 
    def predict_proba(self, X, chunk=2000):
        return np.concatenate([self.forward(normalize(X[i:i + chunk]))[0]
                               for i in range(0, len(X), chunk)], axis=1)
 
    def predict(self, X):
        return self.predict_proba(X).argmax(axis=0)
 
 
def evaluate(model, X, labels):
    probs = model.predict_proba(X)
    return cross_entropy(probs, labels), float(np.mean(probs.argmax(axis=0) == labels))
 
 
def cosine_lr(progress, lr=LEARNING_RATE):
    return lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
 
 
def train(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE,
          use_augmentation=AUGMENT, time_limit=250, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(X_train)
    total_steps = epochs * int(np.ceil(n / batch_size))
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    step, start = 0, time.time()
 
    for epoch in range(epochs):
        order = rng.permutation(n)
        loss_sum, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            x = X_train[idx]
            if use_augmentation:
                x = augment(x, rng)
 
            progress = max(step / total_steps, (time.time() - start) / time_limit)
            a2, cache = model.forward(normalize(x))
            grads = model.backward(a2, one_hot_encode(y_train[idx]), cache)
            model.update(grads, cosine_lr(progress))
            step += 1
 
            loss_sum += cross_entropy(a2, y_train[idx]) * len(idx)
            correct += int((a2.argmax(axis=0) == y_train[idx]).sum())
 
        train_loss, train_acc = loss_sum / n, correct / n
        val_loss, val_acc = evaluate(model, X_val, y_val)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(f"Epoch {epoch + 1:3d}/{epochs} | lr {cosine_lr(progress):.4f} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - start:.0f}s")
 
        if progress >= 1.0:
            print(f"Time limit of {time_limit}s reached, stopping after epoch {epoch + 1}.")
            break
    return history
 
 
def save_csv(predictions, filename="submission.csv"):
    with open(filename, "w") as f:
        f.write("ID,target\n")
        for i, p in enumerate(predictions):
            f.write(f"{i},{p}\n")
    print(f"Created {filename}")
 
 
def main():
    start = time.time()
 
    rng = np.random.default_rng(SEED)
    X, y, X_test = find_data()
 
    order = recover_pixel_order(X)
    X, X_test = deskew(X[:, order]), deskew(X_test[:, order])
    print(f"pixel order recovered + deskew: {time.time() - start:.0f}s")
 
    X_train, y_train, X_val, y_val = split_data(X, y, VAL_SIZE, rng)
    set_normalization(X_train)
    print(f"train {X_train.shape}, validation {X_val.shape}, test {X_test.shape}")
 
    model = MLP()
    train(model, X_train, y_train, X_val, y_val, time_limit=TOTAL_BUDGET - (time.time() - start) - 15)
 
    final_loss, final_acc = evaluate(model, X_train, y_train)
    print(f"Final train (no augmentation): loss {final_loss:.4f} acc {final_acc:.4f}")
 
    save_csv(model.predict(X_test))
    print(f"Total runtime: {time.time() - start:.0f}s")
 
 
if __name__ == "__main__":
    main()