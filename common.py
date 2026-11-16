import math
import numpy as np
import os

def load_and_split_data():
    data_path = os.path.join("data", "records.csv")
    lines = open(data_path, "r").read().strip().split("\n")
    rows = [line.split(",") for line in lines[1:]]

    X_raw = np.array([[float(r[j]) for j in range(1, 8)] for r in rows])
    y_raw = np.array([int(r[8]) for r in rows])
    val_raw = np.array([float(r[9]) for r in rows])
    n_total = len(rows)

    idx_0 = [i for i in range(n_total) if y_raw[i] == 0]
    idx_1 = [i for i in range(n_total) if y_raw[i] == 1]

    n0_train = (7 * len(idx_0)) // 10
    n1_train = (7 * len(idx_1)) // 10

    train_idx = sorted(idx_0[:n0_train] + idx_1[:n1_train])
    test_idx = sorted(idx_0[n0_train:] + idx_1[n1_train:])

    X_tr_raw = X_raw[train_idx]
    y_tr = y_raw[train_idx]
    X_te_raw = X_raw[test_idx]
    y_te = y_raw[test_idx]

    mu_tr = np.mean(X_tr_raw, axis=0)
    sd_tr = np.std(X_tr_raw, axis=0, ddof=1)

    X_tr = (X_tr_raw - mu_tr) / sd_tr
    X_te = (X_te_raw - mu_tr) / sd_tr

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "X_tr_raw": X_tr_raw,
        "y_tr": y_tr,
        "X_te_raw": X_te_raw,
        "y_te": y_te,
        "X_tr": X_tr,
        "X_te": X_te,
        "mu_tr": mu_tr,
        "sd_tr": sd_tr,
        "y_raw": y_raw,
        "value": val_raw
    }

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(X_tr, y_tr, eta=0.1, epochs=1000):
    n = len(y_tr)
    w = np.zeros(X_tr.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = X_tr @ w + b
        p = sigmoid(z)
        gw = (1.0 / n) * (X_tr.T @ (p - y_tr))
        gb = (1.0 / n) * np.sum(p - y_tr)
        w -= eta * gw
        b -= eta * gb
    return w, b
