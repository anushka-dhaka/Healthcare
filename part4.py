import math
import numpy as np
import os
from common import load_and_split_data, sigmoid, train_logistic

# load data
d = load_and_split_data()
X_tr_raw = d["X_tr_raw"]
y_tr = d["y_tr"]
X_te_raw = d["X_te_raw"]
y_te = d["y_te"]
X_tr = d["X_tr"]
X_te = d["X_te"]
n_tr = len(y_tr)
n_te = len(y_te)

# task 4.1: candidate splits
def calc_entropy(pos, total):
    if total == 0 or pos == 0 or pos == total:
        return 0.0
    p1 = pos / total
    p0 = 1.0 - p1
    return -(p1 * math.log2(p1) + p0 * math.log2(p0))

def calc_gini(pos, total):
    if total == 0:
        return 0.0
    p1 = pos / total
    p0 = 1.0 - p1
    return 1.0 - (p1**2 + p0**2)

root_pos = int(np.sum(y_tr))
root_entropy = calc_entropy(root_pos, n_tr)
root_gini = calc_gini(root_pos, n_tr)

# splits conditions on training:
# S1: X6 == 1 (index 5)
# S2: X7 == 1 (index 6)
# S3: X3 > 0.5 (index 2)
# S4: X1 <= 24 (index 0)
splits_cond = [
    ("S1", X_tr_raw[:, 5] == 1),
    ("S2", X_tr_raw[:, 6] == 1),
    ("S3", X_tr_raw[:, 2] > 0.5),
    ("S4", X_tr_raw[:, 0] <= 24)
]

split_records = []
best_info_gain = -1.0
best_split_info = ""
best_gini_gain = -1.0
best_split_gini = ""

for s_name, yes_mask in splits_cond:
    n_yes = int(np.sum(yes_mask))
    pos_yes = int(np.sum(y_tr[yes_mask]))
    n_no = n_tr - n_yes
    pos_no = root_pos - pos_yes
    
    ent_yes = calc_entropy(pos_yes, n_yes)
    ent_no = calc_entropy(pos_no, n_no)
    w_ent = (n_yes / n_tr) * ent_yes + (n_no / n_tr) * ent_no
    info_gain = root_entropy - w_ent
    
    g_yes = calc_gini(pos_yes, n_yes)
    g_no = calc_gini(pos_no, n_no)
    w_gini = (n_yes / n_tr) * g_yes + (n_no / n_tr) * g_no
    gini_gain = root_gini - w_gini
    
    split_records.append((s_name, n_yes, pos_yes, n_no, pos_no, info_gain, gini_gain))
    
    if info_gain > best_info_gain:
        best_info_gain = info_gain
        best_split_info = s_name
    if gini_gain > best_gini_gain:
        best_gini_gain = gini_gain
        best_split_gini = s_name

# task 4.2: stump & model combinations
# stump uses best split by info gain
stump_split_name = best_split_info
# find condition for best split
if stump_split_name == "S1":
    stump_cond_tr = (X_tr_raw[:, 5] == 1)
    stump_cond_te = (X_te_raw[:, 5] == 1)
elif stump_split_name == "S2":
    stump_cond_tr = (X_tr_raw[:, 6] == 1)
    stump_cond_te = (X_te_raw[:, 6] == 1)
elif stump_split_name == "S3":
    stump_cond_tr = (X_tr_raw[:, 2] > 0.5)
    stump_cond_te = (X_te_raw[:, 2] > 0.5)
else:
    stump_cond_tr = (X_tr_raw[:, 0] <= 24)
    stump_cond_te = (X_te_raw[:, 0] <= 24)

n_yes_tr = np.sum(stump_cond_tr)
pos_yes_tr = np.sum(y_tr[stump_cond_tr])
n_no_tr = n_tr - n_yes_tr
pos_no_tr = root_pos - pos_yes_tr

stump_rate_yes = pos_yes_tr / n_yes_tr
stump_rate_no = pos_no_tr / n_no_tr

# predictions on test rows
# 1. logistic (tau = 0.3)
w_log, b_log = train_logistic(X_tr, y_tr, eta=0.1, epochs=1000)
log_prob = sigmoid(X_te @ w_log + b_log)
log_pred = (log_prob >= 0.3).astype(int)

# 2. knn (k = 5)
knn_k = 5
diff_te = X_te[:, np.newaxis, :] - X_tr[np.newaxis, :, :]
dists_te = np.sqrt(np.sum(diff_te**2, axis=2))
sorted_idx_te = np.argsort(dists_te, axis=1, kind="stable")
top_k_idx = sorted_idx_te[:, :knn_k]
top_k_lbls = y_tr[top_k_idx]
knn_prob = np.sum(top_k_lbls, axis=1) / float(knn_k)
knn_pred = (knn_prob >= (knn_k + 1) / (2.0 * knn_k)).astype(int)

# 3. stump (tau = 0.3)
stump_prob = np.where(stump_cond_te, stump_rate_yes, stump_rate_no)
stump_pred = (stump_prob >= 0.3).astype(int)

# 4. hard vote (at least 2 positive)
hard_vote_pred = ((log_pred + knn_pred + stump_pred) >= 2).astype(int)

# 5. soft vote (mean prob >= 0.3)
soft_prob = (log_prob + knn_prob + stump_prob) / 3.0
soft_vote_pred = (soft_prob >= 0.3).astype(int)

def eval_preds(preds):
    tp = int(np.sum((preds == 1) & (y_te == 1)))
    fp = int(np.sum((preds == 1) & (y_te == 0)))
    tn = int(np.sum((preds == 0) & (y_te == 0)))
    fn = int(np.sum((preds == 0) & (y_te == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, tn, fn, prec, rec, f1

models_eval = [
    ("logistic", eval_preds(log_pred)),
    ("knn", eval_preds(knn_pred)),
    ("stump", eval_preds(stump_pred)),
    ("hard_vote", eval_preds(hard_vote_pred)),
    ("soft_vote", eval_preds(soft_vote_pred))
]

# task 4.3: adaboost by hand (2 rounds)
pts = list(range(1, 9))
y_pts = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

# round 1
w_before_1 = np.array([1.0 / 8.0] * 8)
err1 = np.sum(w_before_1[h1 != y_pts])
alpha1 = 0.5 * math.log((1.0 - err1) / err1)
w_unnorm_1 = w_before_1 * np.exp(-alpha1 * y_pts * h1)
w_after_1 = w_unnorm_1 / np.sum(w_unnorm_1)

# round 2
w_before_2 = w_after_1.copy()
err2 = np.sum(w_before_2[h2 != y_pts])
alpha2 = 0.5 * math.log((1.0 - err2) / err2)
w_unnorm_2 = w_before_2 * np.exp(-alpha2 * y_pts * h2)
w_after_2 = w_unnorm_2 / np.sum(w_unnorm_2)

# ensemble accuracy
ens_scores = alpha1 * h1 + alpha2 * h2
ens_preds = np.sign(ens_scores)
ens_acc = np.mean(ens_preds == y_pts)

# adaboost rows for csv
ada_rows = []
for i in range(8):
    ada_rows.append((1, i + 1, w_before_1[i], w_unnorm_1[i], w_after_1[i]))
for i in range(8):
    ada_rows.append((2, i + 1, w_before_2[i], w_unnorm_2[i], w_after_2[i]))

# task 4.4: oob fraction
oob_frac = (1.0 - 1.0 / 560.0)**560

# task 4.5: neural net parameter sizing
# model A option 1 (MLP)
mlp_h1 = 784 * 16 + 16
mlp_h2 = 16 * 16 + 16
mlp_out = 16 * 10 + 10
mlp_tot = mlp_h1 + mlp_h2 + mlp_out

# model A option 2 (CNN)
cnn_c1 = (5 * 5 * 1 + 1) * 6
cnn_c2 = (5 * 5 * 6 + 1) * 16
cnn_d1 = (5 * 5 * 16) * 32 + 32
cnn_d2 = 32 * 10 + 10
cnn_tot = cnn_c1 + cnn_c2 + cnn_d1 + cnn_d2

arch_rows = [
    ("mlp_hidden1", mlp_h1),
    ("mlp_hidden2", mlp_h2),
    ("mlp_output", mlp_out),
    ("cnn_conv1", cnn_c1),
    ("cnn_conv2", cnn_c2),
    ("cnn_dense1", cnn_d1),
    ("cnn_dense2", cnn_d2)
]

fc_layer_params = (128 * 128 * 3) * 1024 + 1024
conv_layer_params = (3 * 3 * 3 + 1) * 64

# model B (RNN)
# W_xh: 128*64, W_hh: 128*128, b_h: 128, W_hy: 16*128, b_y: 16
rnn_tot = 128 * 64 + 128 * 128 + 128 + 16 * 128 + 16

# save part4_results.txt
out_file = os.path.join("outputs", "part4_results.txt")
with open(out_file, "w") as f:
    f.write(f"root_entropy = {root_entropy:.4f}\n")
    f.write(f"root_gini = {root_gini:.4f}\n")
    f.write(f"best_split_by_info_gain = {best_split_info}\n")
    f.write(f"best_split_by_gini_gain = {best_split_gini}\n")
    f.write(f"stump_rate_yes = {stump_rate_yes:.4f}\n")
    f.write(f"stump_rate_no = {stump_rate_no:.4f}\n")
    f.write(f"round1_error = {err1:.4f}\n")
    f.write(f"round1_alpha = {alpha1:.4f}\n")
    f.write(f"round2_error = {err2:.4f}\n")
    f.write(f"round2_alpha = {alpha2:.4f}\n")
    f.write(f"ensemble_accuracy = {ens_acc:.4f}\n")
    f.write(f"oob_fraction = {oob_frac:.4f}\n")
    f.write(f"mlp_total = {mlp_tot}\n")
    f.write(f"cnn_total = {cnn_tot}\n")
    f.write(f"fc_layer_params = {fc_layer_params}\n")
    f.write(f"conv_layer_params = {conv_layer_params}\n")
    f.write(f"rnn_total = {rnn_tot}\n")

# write part4_splits.csv
with open(os.path.join("outputs", "part4_splits.csv"), "w") as f:
    f.write("split,n_yes,positives_yes,n_no,positives_no,info_gain,gini_gain\n")
    for r in split_records:
        f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]:.4f}\n")

# write part4_voting.csv
with open(os.path.join("outputs", "part4_voting.csv"), "w") as f:
    f.write("model,tp,fp,tn,fn,precision,recall,f1\n")
    for name, m in models_eval:
        f.write(f"{name},{m[0]},{m[1]},{m[2]},{m[3]},{m[4]:.4f},{m[5]:.4f},{m[6]:.4f}\n")

# write part4_adaboost.csv
with open(os.path.join("outputs", "part4_adaboost.csv"), "w") as f:
    f.write("round,point,weight_before,weight_unnormalised,weight_after\n")
    for r in ada_rows:
        f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n")

# write part4_architecture.csv
with open(os.path.join("outputs", "part4_architecture.csv"), "w") as f:
    f.write("layer,parameters\n")
    for r in arch_rows:
        f.write(f"{r[0]},{r[1]}\n")

print("Part 4 finished successfully")
