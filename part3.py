import math
import numpy as np
import os

# read records
data_path = os.path.join("data", "records.csv")
lines = open(data_path, "r").read().strip().split("\n")
rows = [line.split(",") for line in lines[1:]]

# X1 to X5 continuous
X_cont = np.array([[float(r[j]) for j in range(1, 6)] for r in rows])
outcome = np.array([int(r[8]) for r in rows])
value = np.array([float(r[9]) for r in rows])
n = len(rows)

# task 3.1: standardize x1-x5 over all 800 rows
mu_x = np.mean(X_cont, axis=0)
sd_x = np.std(X_cont, axis=0, ddof=1)
X_std = (X_cont - mu_x) / sd_x

def run_kmeans(k_val):
    step = 800 // k_val
    init_indices = [i * step for i in range(k_val)]
    centroids = X_std[init_indices].copy()
    labels = np.zeros(n, dtype=int)
    
    while True:
        # dist to each centroid: (n, k)
        diff = X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        
        # tie break: lower numbered centroid
        new_labels = np.argmin(sq_dists, axis=1)
        
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        
        # update centroids
        for c_idx in range(k_val):
            members = X_std[labels == c_idx]
            if len(members) > 0:
                centroids[c_idx] = np.mean(members, axis=0)
                
    # compute wcss
    wcss = 0.0
    for i in range(n):
        wcss += np.sum((X_std[i] - centroids[labels[i]])**2)
    return wcss, labels, centroids

wcss_dict = {}
labels_dict = {}
for k in [2, 3, 4, 5, 6]:
    w, lbls, _ = run_kmeans(k)
    wcss_dict[k] = w
    labels_dict[k] = lbls

# task 3.2: elbow k
# smallest K in {3, 4, 5} for which WCSS(K) - WCSS(K + 1) < 0.5 * (WCSS(K - 1) - WCSS(K))
elbow_k = None
for k in [3, 4, 5]:
    drop_curr = wcss_dict[k] - wcss_dict[k + 1]
    drop_prev = wcss_dict[k - 1] - wcss_dict[k]
    if drop_curr < 0.5 * drop_prev:
        elbow_k = k
        break

# task 3.3: profile k=3 segments
lbl_3 = labels_dict[3]
# order clusters by ascending mean of X3 in original units
mean_x3_per_cluster = []
for c_idx in range(3):
    mean_x3_per_cluster.append(np.mean(X_cont[lbl_3 == c_idx, 2]))

cluster_order = np.argsort(mean_x3_per_cluster)
# cluster_order[0] -> A, [1] -> B, [2] -> C
seg_names = ["A", "B", "C"]
seg_map = {cluster_order[i]: seg_names[i] for i in range(3)}
record_segments = np.array([seg_map[lbl_3[i]] for i in range(n)])

seg_rows = []
seg_sizes = {}
for name in ["A", "B", "C"]:
    mask = (record_segments == name)
    s_size = int(np.sum(mask))
    seg_sizes[name] = s_size
    m_x = np.mean(X_cont[mask], axis=0)
    out_rate = np.mean(outcome[mask])
    m_val = np.mean(value[mask])
    seg_rows.append((name, s_size, m_x[0], m_x[1], m_x[2], m_x[3], m_x[4], out_rate, m_val))

# task 3.4: hierarchical clustering
cohorts = ["E1", "E2", "E3", "E4", "E5", "E6"]
dist_table = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

def run_hierarchical(linkage_type):
    # initial clusters
    clusters = [[i] for i in range(6)]
    merges = []
    
    for m_no in range(1, 6):
        min_d = float("inf")
        merge_pair = None
        
        # find pair with min dist
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                all_dists = [dist_table[p1, p2] for p1 in c1 for p2 in c2]
                if linkage_type == "single":
                    d = min(all_dists)
                elif linkage_type == "complete":
                    d = max(all_dists)
                elif linkage_type == "average":
                    d = sum(all_dists) / len(all_dists)
                
                if d < min_d:
                    min_d = d
                    merge_pair = (i, j)
                    
        i, j = merge_pair
        c1, c2 = clusters[i], clusters[j]
        # format names
        c1_members = sorted(c1)
        c2_members = sorted(c2)
        # cluster_a is whose smallest member is smaller
        if c1_members[0] > c2_members[0]:
            c1_members, c2_members = c2_members, c1_members
            
        c_a_str = "+".join(cohorts[idx] for idx in c1_members)
        c_b_str = "+".join(cohorts[idx] for idx in c2_members)
        merges.append((m_no, c_a_str, c_b_str, min_d))
        
        # update cluster list
        new_c = sorted(c1 + c2)
        clusters = [clusters[idx] for idx in range(len(clusters)) if idx not in (i, j)]
        clusters.append(new_c)
        
    # dendrogram cut: largest gap between consecutive merge heights
    heights = [m[3] for m in merges]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    max_gap_idx = int(np.argmax(gaps)) # 0 means between merge 1 & 2 -> 1 merge below cut
    merges_below = max_gap_idx + 1
    clusters_at_cut = 6 - merges_below
    return merges, clusters_at_cut

single_merges, single_cut = run_hierarchical("single")
complete_merges, complete_cut = run_hierarchical("complete")
average_merges, average_cut = run_hierarchical("average")

# task 3.5: pca
S = (1.0 / (n - 1)) * (X_std.T @ X_std)
eigenvalues, eigenvectors = np.linalg.eigh(S)

# sort descending
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sort_idx]
eigenvectors = eigenvectors[:, sort_idx]

# sign convention: flip eigenvector so entry of largest abs value is positive
for j in range(5):
    vec = eigenvectors[:, j]
    max_idx = np.argmax(np.abs(vec))
    if vec[max_idx] < 0:
        eigenvectors[:, j] = -vec

eigenvalue_sum = np.sum(eigenvalues)
explained_ratio = eigenvalues / eigenvalue_sum
cumulative_ratio = np.cumsum(explained_ratio)

components_for_80pct = int(np.where(cumulative_ratio >= 0.8)[0][0] + 1)

W2 = eigenvectors[:, :2]
Z = X_std @ W2

# segment mean pc1
mean_pc1_A = np.mean(Z[record_segments == "A", 0])
mean_pc1_B = np.mean(Z[record_segments == "B", 0])
mean_pc1_C = np.mean(Z[record_segments == "C", 0])

# save part3_results.txt
out_file = os.path.join("outputs", "part3_results.txt")
with open(out_file, "w") as f:
    f.write(f"wcss_k2 = {wcss_dict[2]:.4f}\n")
    f.write(f"wcss_k3 = {wcss_dict[3]:.4f}\n")
    f.write(f"wcss_k4 = {wcss_dict[4]:.4f}\n")
    f.write(f"wcss_k5 = {wcss_dict[5]:.4f}\n")
    f.write(f"wcss_k6 = {wcss_dict[6]:.4f}\n")
    f.write(f"elbow_k = {elbow_k}\n")
    f.write(f"size_segment_A = {seg_sizes['A']}\n")
    f.write(f"size_segment_B = {seg_sizes['B']}\n")
    f.write(f"size_segment_C = {seg_sizes['C']}\n")
    f.write(f"single_clusters_at_cut = {single_cut}\n")
    f.write(f"complete_clusters_at_cut = {complete_cut}\n")
    f.write(f"average_clusters_at_cut = {average_cut}\n")
    f.write(f"eigenvalue_sum = {eigenvalue_sum:.4f}\n")
    for j in range(5):
        f.write(f"eigenvalue_{j+1} = {eigenvalues[j]:.4f}\n")
    for j in range(5):
        f.write(f"explained_ratio_{j+1} = {explained_ratio[j]:.4f}\n")
    for j in range(5):
        f.write(f"cumulative_ratio_{j+1} = {cumulative_ratio[j]:.4f}\n")
    f.write(f"components_for_80pct = {components_for_80pct}\n")
    for j in range(5):
        f.write(f"pc1_loading_X{j+1} = {eigenvectors[j, 0]:.4f}\n")
    for j in range(5):
        f.write(f"pc2_loading_X{j+1} = {eigenvectors[j, 1]:.4f}\n")
    f.write(f"row1_pc1 = {Z[0, 0]:.4f}\n")
    f.write(f"row1_pc2 = {Z[0, 1]:.4f}\n")
    f.write(f"row2_pc1 = {Z[1, 0]:.4f}\n")
    f.write(f"row2_pc2 = {Z[1, 1]:.4f}\n")
    f.write(f"row3_pc1 = {Z[2, 0]:.4f}\n")
    f.write(f"row3_pc2 = {Z[2, 1]:.4f}\n")
    f.write(f"mean_pc1_segment_A = {mean_pc1_A:.4f}\n")
    f.write(f"mean_pc1_segment_B = {mean_pc1_B:.4f}\n")
    f.write(f"mean_pc1_segment_C = {mean_pc1_C:.4f}\n")

# write part3_segments.csv
with open(os.path.join("outputs", "part3_segments.csv"), "w") as f:
    f.write("segment,size,mean_X1,mean_X2,mean_X3,mean_X4,mean_X5,outcome_rate,mean_value\n")
    for row in seg_rows:
        f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},{row[6]:.4f},{row[7]:.4f},{row[8]:.4f}\n")

# write part3_hierarchical.csv
with open(os.path.join("outputs", "part3_hierarchical.csv"), "w") as f:
    f.write("linkage,merge_no,cluster_a,cluster_b,height\n")
    for m in single_merges:
        f.write(f"single,{m[0]},{m[1]},{m[2]},{m[3]:.4f}\n")
    for m in complete_merges:
        f.write(f"complete,{m[0]},{m[1]},{m[2]},{m[3]:.4f}\n")
    for m in average_merges:
        f.write(f"average,{m[0]},{m[1]},{m[2]},{m[3]:.4f}\n")

print("Part 3 finished successfully")
