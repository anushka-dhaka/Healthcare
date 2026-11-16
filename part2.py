import math
import numpy as np
import os
from common import load_and_split_data, sigmoid

# load & split
d = load_and_split_data()
train_idx, test_idx = d["train_idx"], d["test_idx"]
X_tr, y_tr = d["X_tr"], d["y_tr"]
X_te, y_te = d["X_te"], d["y_te"]

train_rows = len(train_idx)
test_rows = len(test_idx)
train_positives = int(np.sum(y_tr))
test_positives = int(np.sum(y_te))

# task 2.2: logistic reg from scratch
w = np.zeros(7)
b = 0.0
eta = 0.1

def bce_loss(X_mat, y_vec, w_curr, b_curr):
    z = X_mat @ w_curr + b_curr
    p = sigmoid(z)
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y_vec * np.log(p) + (1.0 - y_vec) * np.log(1.0 - p))

loss_epoch_0 = bce_loss(X_tr, y_tr, w, b)
loss_epoch_1 = 0.0
loss_epoch_10 = 0.0
loss_epoch_100 = 0.0
loss_epoch_1000 = 0.0

for ep in range(1, 1001):
    z = X_tr @ w + b
    p = sigmoid(z)
    grad_w = (1.0 / train_rows) * (X_tr.T @ (p - y_tr))
    grad_b = (1.0 / train_rows) * np.sum(p - y_tr)
    w -= eta * grad_w
    b -= eta * grad_b
    if ep == 1:
        loss_epoch_1 = bce_loss(X_tr, y_tr, w, b)
    elif ep == 10:
        loss_epoch_10 = bce_loss(X_tr, y_tr, w, b)
    elif ep == 100:
        loss_epoch_100 = bce_loss(X_tr, y_tr, w, b)
    elif ep == 1000:
        loss_epoch_1000 = bce_loss(X_tr, y_tr, w, b)

test_bce = bce_loss(X_te, y_te, w, b)

# task 2.3: lr effect 200 epochs
def train_lr(eta_val):
    w_tmp = np.zeros(7)
    b_tmp = 0.0
    for _ in range(200):
        z_tmp = X_tr @ w_tmp + b_tmp
        p_tmp = sigmoid(z_tmp)
        gw = (1.0 / train_rows) * (X_tr.T @ (p_tmp - y_tr))
        gb = (1.0 / train_rows) * np.sum(p_tmp - y_tr)
        w_tmp -= eta_val * gw
        b_tmp -= eta_val * gb
    return bce_loss(X_tr, y_tr, w_tmp, b_tmp)

loss_epoch_200_eta_001 = train_lr(0.01)
loss_epoch_200_eta_01 = train_lr(0.1)
loss_epoch_200_eta_10 = train_lr(1.0)

# task 2.4: odds ratios
odds_ratios = np.exp(w)

# task 2.5: eval test set at tau=0.5 and 0.3
def calc_metrics(y_true, y_prob, tau):
    y_pred = (y_prob >= tau).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n_sample = len(y_true)
    acc = (tp + tn) / n_sample if n_sample > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tp, fp, tn, fn, acc, prec, rec, f1, spec, fpr

test_prob = sigmoid(X_te @ w + b)
m5 = calc_metrics(y_te, test_prob, 0.5)
m3 = calc_metrics(y_te, test_prob, 0.3)
blind_acc = (test_rows - test_positives) / test_rows

# task 2.6: threshold sweep
sweep_taus = [0.1 * k for k in range(1, 10)]
sweep_rows = []
for tau in sweep_taus:
    tp, fp, tn, fn, acc, prec, rec, f1, spec, fpr = calc_metrics(y_te, test_prob, tau)
    sweep_rows.append((tau, tp, fp, tn, fn, rec, fpr))

# task 2.7: knn with cv
folds = [i % 5 for i in range(train_rows)]
k_values = list(range(1, 23, 2))
cv_f1_means = []

for k in k_values:
    fold_f1s = []
    for f in range(5):
        val_mask = np.array([idx for idx in range(train_rows) if folds[idx] == f])
        ref_mask = np.array([idx for idx in range(train_rows) if folds[idx] != f])
        
        X_ref = X_tr[ref_mask]
        y_ref = y_tr[ref_mask]
        X_val = X_tr[val_mask]
        y_val = y_tr[val_mask]
        
        diff = X_val[:, np.newaxis, :] - X_ref[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        sorted_indices = np.argsort(dists, axis=1, kind="stable")
        top_k_indices = sorted_indices[:, :k]
        top_k_labels = y_ref[top_k_indices]
        
        pos_counts = np.sum(top_k_labels, axis=1)
        preds = (pos_counts >= (k + 1) // 2).astype(int)
        
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fold_f1s.append(f1)
    cv_f1_means.append(np.mean(fold_f1s))

best_idx = int(np.argmax(cv_f1_means))
knn_best_k = k_values[best_idx]

# refit knn
diff_test = X_te[:, np.newaxis, :] - X_tr[np.newaxis, :, :]
dists_test = np.sqrt(np.sum(diff_test**2, axis=2))
sorted_idx_test = np.argsort(dists_test, axis=1, kind="stable")
top_k_test = sorted_idx_test[:, :knn_best_k]
top_k_labels_test = y_tr[top_k_test]
knn_test_pos = np.sum(top_k_labels_test, axis=1)
knn_test_preds = (knn_test_pos >= (knn_best_k + 1) // 2).astype(int)

knn_tp = int(np.sum((knn_test_preds == 1) & (y_te == 1)))
knn_fp = int(np.sum((knn_test_preds == 1) & (y_te == 0)))
knn_tn = int(np.sum((knn_test_preds == 0) & (y_te == 0)))
knn_fn = int(np.sum((knn_test_preds == 0) & (y_te == 1)))
knn_acc = (knn_tp + knn_tn) / test_rows
knn_prec = knn_tp / (knn_tp + knn_fp) if (knn_tp + knn_fp) > 0 else 0.0
knn_rec = knn_tp / (knn_tp + knn_fn) if (knn_tp + knn_fn) > 0 else 0.0
knn_f1 = (2 * knn_prec * knn_rec) / (knn_prec + knn_rec) if (knn_prec + knn_rec) > 0 else 0.0

# task 2.8: cost optimization
train_prob = sigmoid(X_tr @ w + b)
best_cost_train = float("inf")
cost_tau = 0.0

for k_tau in range(1, 20):
    tau = k_tau / 20.0
    pred_tr = (train_prob >= tau).astype(int)
    fp = np.sum((pred_tr == 1) & (y_tr == 0))
    fn = np.sum((pred_tr == 0) & (y_tr == 1))
    c_val = 50 * fp + 400 * fn
    if c_val < best_cost_train:
        best_cost_train = c_val
        cost_tau = tau

cost_train_at_tau = int(best_cost_train)

pred_te_tau = (test_prob >= cost_tau).astype(int)
fp_te = np.sum((pred_te_tau == 1) & (y_te == 0))
fn_te = np.sum((pred_te_tau == 0) & (y_te == 1))
cost_test_at_tau = int(50 * fp_te + 400 * fn_te)

pred_te_05 = (test_prob >= 0.5).astype(int)
fp_05 = np.sum((pred_te_05 == 1) & (y_te == 0))
fn_05 = np.sum((pred_te_05 == 0) & (y_te == 1))
cost_test_at_05 = int(50 * fp_05 + 400 * fn_05)

cost_test_flag_nobody = int(400 * test_positives)

# write part2_results.txt
out_file = os.path.join("outputs", "part2_results.txt")
with open(out_file, "w") as f:
    f.write(f"train_rows = {train_rows}\n")
    f.write(f"train_positives = {train_positives}\n")
    f.write(f"test_rows = {test_rows}\n")
    f.write(f"test_positives = {test_positives}\n")
    f.write(f"loss_epoch_0 = {loss_epoch_0:.4f}\n")
    f.write(f"loss_epoch_1 = {loss_epoch_1:.4f}\n")
    f.write(f"loss_epoch_10 = {loss_epoch_10:.4f}\n")
    f.write(f"loss_epoch_100 = {loss_epoch_100:.4f}\n")
    f.write(f"loss_epoch_1000 = {loss_epoch_1000:.4f}\n")
    for j in range(7):
        f.write(f"weight_X{j+1} = {w[j]:.4f}\n")
    f.write(f"bias = {b:.4f}\n")
    f.write(f"test_bce = {test_bce:.4f}\n")
    f.write(f"loss_epoch_200_eta_0.01 = {loss_epoch_200_eta_001:.4f}\n")
    f.write(f"loss_epoch_200_eta_0.1 = {loss_epoch_200_eta_01:.4f}\n")
    f.write(f"loss_epoch_200_eta_1.0 = {loss_epoch_200_eta_10:.4f}\n")
    for j in range(7):
        f.write(f"odds_ratio_X{j+1} = {odds_ratios[j]:.4f}\n")
    f.write(f"tau_0.5_tp = {m5[0]}\n")
    f.write(f"tau_0.5_fp = {m5[1]}\n")
    f.write(f"tau_0.5_tn = {m5[2]}\n")
    f.write(f"tau_0.5_fn = {m5[3]}\n")
    f.write(f"tau_0.5_accuracy = {m5[4]:.4f}\n")
    f.write(f"tau_0.5_precision = {m5[5]:.4f}\n")
    f.write(f"tau_0.5_recall = {m5[6]:.4f}\n")
    f.write(f"tau_0.5_f1 = {m5[7]:.4f}\n")
    f.write(f"tau_0.5_specificity = {m5[8]:.4f}\n")
    f.write(f"tau_0.3_tp = {m3[0]}\n")
    f.write(f"tau_0.3_fp = {m3[1]}\n")
    f.write(f"tau_0.3_tn = {m3[2]}\n")
    f.write(f"tau_0.3_fn = {m3[3]}\n")
    f.write(f"tau_0.3_accuracy = {m3[4]:.4f}\n")
    f.write(f"tau_0.3_precision = {m3[5]:.4f}\n")
    f.write(f"tau_0.3_recall = {m3[6]:.4f}\n")
    f.write(f"tau_0.3_f1 = {m3[7]:.4f}\n")
    f.write(f"tau_0.3_specificity = {m3[8]:.4f}\n")
    f.write(f"blind_accuracy = {blind_acc:.4f}\n")
    f.write(f"knn_best_k = {knn_best_k}\n")
    f.write(f"knn_tp = {knn_tp}\n")
    f.write(f"knn_fp = {knn_fp}\n")
    f.write(f"knn_tn = {knn_tn}\n")
    f.write(f"knn_fn = {knn_fn}\n")
    f.write(f"knn_accuracy = {knn_acc:.4f}\n")
    f.write(f"knn_precision = {knn_prec:.4f}\n")
    f.write(f"knn_recall = {knn_rec:.4f}\n")
    f.write(f"knn_f1 = {knn_f1:.4f}\n")
    f.write(f"cost_tau = {cost_tau:.4f}\n")
    f.write(f"cost_train_at_tau = {cost_train_at_tau}\n")
    f.write(f"cost_test_at_tau = {cost_test_at_tau}\n")
    f.write(f"cost_test_at_0.5 = {cost_test_at_05}\n")
    f.write(f"cost_test_flag_nobody = {cost_test_flag_nobody}\n")

# write part2_thresholds.csv
with open(os.path.join("outputs", "part2_thresholds.csv"), "w") as f:
    f.write("tau,tp,fp,tn,fn,tpr,fpr\n")
    for row in sweep_rows:
        f.write(f"{row[0]:.4f},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]:.4f},{row[6]:.4f}\n")

# write part2_knn_cv.csv
with open(os.path.join("outputs", "part2_knn_cv.csv"), "w") as f:
    f.write("K,mean_cv_f1\n")
    for k, mean_f1 in zip(k_values, cv_f1_means):
        f.write(f"{k},{mean_f1:.4f}\n")

print("Part 2 finished successfully")
