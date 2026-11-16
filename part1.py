import math
import numpy as np
import os

# read records
data_path = os.path.join("data", "records.csv")
lines = open(data_path, "r").read().strip().split("\n")
header = lines[0].split(",")
rows = [line.split(",") for line in lines[1:]]

# cols: 0:id, 1:x1, 2:x2, 3:x3, 4:x4, 5:x5, 6:x6, 7:x7, 8:outcome, 9:value
X1 = np.array([float(r[1]) for r in rows])
X2 = np.array([float(r[2]) for r in rows])
X3 = np.array([float(r[3]) for r in rows])
X4 = np.array([float(r[4]) for r in rows])
X5 = np.array([float(r[5]) for r in rows])
X6 = np.array([float(r[6]) for r in rows])
X7 = np.array([float(r[7]) for r in rows])
outcome = np.array([int(r[8]) for r in rows])
value = np.array([float(r[9]) for r in rows])

n = len(value)

# task 1.1: stats & ci
mean_val = np.mean(value)
sd_val = np.std(value, ddof=1)
se_val = sd_val / math.sqrt(n)
ci95_low = mean_val - 1.96 * se_val
ci95_high = mean_val + 1.96 * se_val

# task 1.2: hyp test
z_stat = (mean_val - 8.0) / se_val
reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

# task 1.3: sample size
E = 0.5
n_req = int(math.ceil((1.96 * sd_val / E)**2))

# task 1.4: closed form lin reg on x4
cov_x4_val = np.sum((X4 - np.mean(X4)) * (value - mean_val)) / (n - 1)
var_x4 = np.var(X4, ddof=1)
reg_slope = cov_x4_val / var_x4
reg_intercept = mean_val - reg_slope * np.mean(X4)

fitted = reg_intercept + reg_slope * X4
tss = np.sum((value - mean_val)**2)
ess = np.sum((fitted - mean_val)**2)
rss = np.sum((value - fitted)**2)
tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
r2 = 1.0 - (rss / tss)
rmse = math.sqrt(rss / n)
mae = np.mean(np.abs(value - fitted))

# task 1.5: gd fit
x4_sd = np.std(X4, ddof=1)
z = (X4 - np.mean(X4)) / x4_sd
m, c = 0.0, 0.0
eta = 0.1

def cost_fn(m_curr, c_curr):
    y_pred = m_curr * z + c_curr
    return (1.0 / n) * np.sum((value - y_pred)**2)

gd_J0 = cost_fn(m, c)
gd_J1 = 0.0
gd_J10 = 0.0
gd_J100 = 0.0
gd_J500 = 0.0

for step in range(1, 501):
    e = value - (m * z + c)
    grad_m = -(2.0 / n) * np.sum(z * e)
    grad_c = -(2.0 / n) * np.sum(e)
    m -= eta * grad_m
    c -= eta * grad_c
    if step == 1:
        gd_J1 = cost_fn(m, c)
    elif step == 10:
        gd_J10 = cost_fn(m, c)
    elif step == 100:
        gd_J100 = cost_fn(m, c)
    elif step == 500:
        gd_J500 = cost_fn(m, c)

gd_slope = m / x4_sd
gd_intercept = c - m * np.mean(X4) / x4_sd
gd_slope_gap = abs(gd_slope - reg_slope)

# task 1.6: funnel check & log transform
raw_order = np.argsort(fitted, kind="stable")
raw_resids = value - fitted
lower_raw = raw_resids[raw_order[:400]]
upper_raw = raw_resids[raw_order[400:]]
resid_sd_lower_raw = np.std(lower_raw, ddof=1)
resid_sd_upper_raw = np.std(upper_raw, ddof=1)
resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

# log log reg
log_val = np.log(value)
log_x4 = np.log(X4)
mean_log_val = np.mean(log_val)
mean_log_x4 = np.mean(log_x4)
cov_log = np.sum((log_x4 - mean_log_x4) * (log_val - mean_log_val)) / (n - 1)
var_log = np.var(log_x4, ddof=1)
loglog_slope = cov_log / var_log
loglog_intercept = mean_log_val - loglog_slope * mean_log_x4
log_fitted = loglog_intercept + loglog_slope * log_x4

log_tss = np.sum((log_val - mean_log_val)**2)
log_rss = np.sum((log_val - log_fitted)**2)
loglog_r2 = 1.0 - (log_rss / log_tss)
loglog_rmse = math.sqrt(log_rss / n)

log_order = np.argsort(log_fitted, kind="stable")
log_resids = log_val - log_fitted
lower_log = log_resids[log_order[:400]]
upper_log = log_resids[log_order[400:]]
resid_sd_lower_log = np.std(lower_log, ddof=1)
resid_sd_upper_log = np.std(upper_log, ddof=1)
resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

# task 1.7: multiple reg
X_mat = np.column_stack([
    np.ones(n),
    log_x4,
    X1,
    X2,
    X3,
    X5,
    X6,
    X7
])
# normal eq solve
beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ log_val)
mr_fitted = X_mat @ beta
mr_rss = np.sum((log_val - mr_fitted)**2)
mr_r2 = 1.0 - (mr_rss / log_tss)
mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

# save results
os.makedirs("outputs", exist_ok=True)
out_file = os.path.join("outputs", "part1_results.txt")
with open(out_file, "w") as f:
    f.write(f"n = {n}\n")
    f.write(f"mean_value = {mean_val:.4f}\n")
    f.write(f"sd_value = {sd_val:.4f}\n")
    f.write(f"se_value = {se_val:.4f}\n")
    f.write(f"ci95_low = {ci95_low:.4f}\n")
    f.write(f"ci95_high = {ci95_high:.4f}\n")
    f.write(f"z_stat = {z_stat:.4f}\n")
    f.write(f"reject_h0 = {reject_h0}\n")
    f.write(f"n_required = {n_req}\n")
    f.write(f"reg_slope = {reg_slope:.4f}\n")
    f.write(f"reg_intercept = {reg_intercept:.4f}\n")
    f.write(f"tss = {tss:.4f}\n")
    f.write(f"ess = {ess:.4f}\n")
    f.write(f"rss = {rss:.4f}\n")
    f.write(f"tss_equals_ess_plus_rss = {tss_equals_ess_plus_rss}\n")
    f.write(f"r2 = {r2:.4f}\n")
    f.write(f"rmse = {rmse:.4f}\n")
    f.write(f"mae = {mae:.4f}\n")
    f.write(f"gd_J0 = {gd_J0:.4f}\n")
    f.write(f"gd_J1 = {gd_J1:.4f}\n")
    f.write(f"gd_J10 = {gd_J10:.4f}\n")
    f.write(f"gd_J100 = {gd_J100:.4f}\n")
    f.write(f"gd_J500 = {gd_J500:.4f}\n")
    f.write(f"gd_slope = {gd_slope:.4f}\n")
    f.write(f"gd_intercept = {gd_intercept:.4f}\n")
    f.write(f"gd_slope_gap = {gd_slope_gap:.4f}\n")
    f.write(f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}\n")
    f.write(f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}\n")
    f.write(f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}\n")
    f.write(f"loglog_intercept = {loglog_intercept:.4f}\n")
    f.write(f"loglog_slope = {loglog_slope:.4f}\n")
    f.write(f"loglog_r2 = {loglog_r2:.4f}\n")
    f.write(f"loglog_rmse = {loglog_rmse:.4f}\n")
    f.write(f"resid_sd_lower_log = {resid_sd_lower_log:.4f}\n")
    f.write(f"resid_sd_upper_log = {resid_sd_upper_log:.4f}\n")
    f.write(f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}\n")
    for idx in range(8):
        f.write(f"mr_beta_{idx} = {beta[idx]:.4f}\n")
    f.write(f"mr_r2 = {mr_r2:.4f}\n")
    f.write(f"mr_adj_r2 = {mr_adj_r2:.4f}\n")
    f.write(f"loglog_adj_r2 = {loglog_adj_r2:.4f}\n")

print("Part 1 finished successfully")
