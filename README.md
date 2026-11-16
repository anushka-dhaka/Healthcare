# Healthcare
pharmeasy capstone project 
Domain: Healthcare

## how to run

everything runs from the repository root using standard Python (3.10+) and NumPy.
Clone the repo and run the following in the terminal:
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
All generated results and CSV files will be written to outputs/.

>>>>> ## part 1 findings
- R1.1: The null hypothesis H0 (μ = 8.0) is rejected at α = 0.05 because the test statistic z_stat = 2.8930 exceeds the two-tailed critical threshold of 1.96. In domain terms, a Type I error would occur if PharmEasy concluded that the true mean care-plan cost had drifted away from the budgeted INR 8,000 baseline when in fact it had not, leading to unnecessary budget revisions. The estimated gap mean_value − 8.0 is 0.9382 column units (INR 938.20 per patient), which represents an ~11.7% cost overrun above baseline and is practically significant for annual operational budgeting.
- R1.2: Sizing a 95% confidence interval with margin of error E = INR 500 (0.5 units) requires n_required = 1293 samples. Because only 800 patient records are currently available, the planning team's existing sample is underpowered by 493 records to guarantee the target precision.
- R1.3: The difference between the closed-form regression slope and the gradient descent slope is gd_slope_gap = 0.0000. Gradient descent reached the exact analytical line because ordinary least squares regression defines a strictly convex parabolic loss surface with a single global minimum, and feature standardisation with learning rate η = 0.1 enabled stable convergence within 500 iterations.
- R1.4: In the raw regression model, resid_sd_ratio_raw is 3.7525 (resid_sd_lower_raw = 2.3155, resid_sd_upper_raw = 8.6891), clearly displaying a funnel effect where residual variance expands dramatically as bill amounts grow. In the log-log specification, resid_sd_ratio_log drops to 1.0396 (resid_sd_lower_log = 0.5165, resid_sd_upper_log = 0.5369), confirming that the logarithmic transformation stabilizes the variance and eliminates the funnel.
- R1.5: In multiple regression Task 1.7, the predictors with absolute coefficients above 0.05 (excluding the constant) are ln(X4) (1.1576), X2 (0.0940), X3 (0.1067), and X6 (0.1518). Comparing mr_adj_r2 = 0.7403 against the single-predictor loglog_adj_r2 = 0.6561 shows an 8.4 percentage-point gain in explained variance, indicating that adding tenure, friction events, copay burden, and chronic condition flags earns its place.

>>>>>>>>## part 2 findings
- R2.1: The training losses after 200 epochs ordered from lowest to highest are loss_epoch_200_eta_1.0 (0.3703) < loss_epoch_200_eta_0.1 (0.3776) < loss_epoch_200_eta_0.01 (0.5091). A learning rate that is too small (0.01) makes sluggish progress across 200 iterations, whereas larger learning rates (0.1 and 1.0) descend the convex surface rapidly without overshooting.
- R2.2: The three predictors with the largest absolute weights in logistic regression are weight_X3 = 0.9013 (copay_burden_ratio), weight_X1 = -0.5554 (months_registered), and weight_X2 = 0.5505 (missed_visit_count_12m). The largest odds ratio is odds_ratio_X3 = 2.4629: each one-standard-deviation rise in copay burden ratio multiplies a patient's odds of missing an appointment by approximately 2.46, holding other predictors constant.
- R2.3: The baseline blind_accuracy is 0.7875, while the logistic model at τ = 0.5 achieves tau_0.5_accuracy = 0.8042. Accuracy alone is misleading because the dataset is class-imbalanced (~21% positives), meaning a trivial model predicting zero for everyone already scores nearly 79% while missing every single no-show. Lowering τ from 0.5 to 0.3 increased recall from 0.4706 to 0.7451 (capturing 38 vs 24 actual no-shows) while precision decreased from 0.5455 to 0.4634.
- R2.4: At τ = 0.3, the logistic regression model achieved tau_0.3_f1 = 0.5714 and tau_0.3_recall = 0.7451, whereas the 5-nearest-neighbour model reached knn_f1 = 0.3696 and knn_recall = 0.3333. Logistic regression clearly outperforms KNN on both detection coverage and balanced harmonic performance.
- R2.5: Deploy the logistic regression model at threshold τ = 0.1500 (cost_tau). This threshold achieves a test cost of INR 6,050 (cost_test_at_tau), which cuts the cost nearly in half compared to INR 11,800 at τ = 0.5 (cost_test_at_0.5) and saves INR 14,350 relative to the INR 20,400 cost of flagging nobody (cost_test_flag_nobody). The threshold was selected on training data rather than test data to ensure an honest, un-overfitted operational cost estimate on unseen records.

>>>>>>>>## Part 3 findings
- R3.1: The successive WCSS drops are:
  - WCSS(2) − WCSS(3) = 659.8282
  - WCSS(3) − WCSS(4) = 108.0739
  - WCSS(4) − WCSS(5) = 171.0575
  - WCSS(5) − WCSS(6) = 45.5466
  The elbow test flags elbow_k = 3 because the drop from 3 to 4 (108.0739) is less than half the drop from 2 to 3 (329.9141). The subsequent larger drop from 4 to 5 occurs because the deterministic starting rows provide a pinned, reproducible initialization rather than guaranteed global optima at each K.
- R3.2: Segment profiles from part3_segments.csv:
  - Segment A (size 322): Established Low-Risk Members (mean tenure 47.5 months, adherence 62.2, copay ratio 0.2250, care-plan cost INR 3,598). Operational action: Automate self-serve portal check-ins and standard SMS confirmations without expensive manual intervention.
  - Segment B (size 255): High-Value Complex Care (mean bill INR 28,719, care-plan cost INR 17,125, missed visits 6.07, no-show rate 0.2353). Operational action: Assign dedicated teleconsultation relationship managers and VIP appointment booking to safeguard recurring revenue.
  - Segment C (size 223): High-Friction Novices (mean tenure 14.4 months, copay ratio 0.7118, adherence 38.5, no-show rate 0.4664, care-plan cost INR 7,288). Operational action: Provide proactive financial counselling, split copay payment options, and live telephone reminders 24 hours prior to slot.
  Segment C has the highest outcome_rate at 0.4664.
- R3.3: At the widest merge gap, single linkage yields 4 clusters ({E2, E3, E6}, {E1}, {E4}, {E5}), whereas complete linkage yields 2 clusters ({E1, E4, E5} and {E2, E3, E6}). The two rules disagree because single linkage focuses solely on nearest neighbour pairs, causing chaining that merges E2, E3, and E6 while stranding other points as singletons, whereas complete linkage evaluates maximum cluster diameter, forming compact, balanced groups.
- R3.4: The first principal component (PC1) explains 52.27% of variance and is driven in the positive direction by tenure X1 (0.5317) and adherence X5 (0.4418), and in the negative direction by copay burden X3 (-0.5187) and missed visits X2 (-0.4100). Retaining at least 80% of total variance requires components_for_80pct = 3 components (cumulative ratio = 86.16%). The mean PC1 scores across segments are +1.7773 for Segment A, -0.7831 for Segment B, and -1.6709 for Segment C, proving that PC1 separates stable, adherent patients from vulnerable, high-friction cohorts.

>>>>>## part 4 findings
- R4.1: Candidate split S4 (months_registered ≤ 24) is the best split because it achieves the highest information gain (0.1252) and Gini gain (0.0606), isolating recent registrants whose no-show rate is 45.74% compared to 8.87% for longer-tenured patients. Growing a tree until every leaf is pure causes severe overfitting by memorizing random noise and idiosyncrasies in the training set. Setting a maximum depth limit (max_depth) or minimum leaf size stops tree growth before overfitting occurs.
- R4.2: We recommend shipping the standalone logistic regression model at τ = 0.3. It yields the top F1 score of 0.5714 with strong recall of 0.7451. Combining models did not improve performance: hard voting dropped F1 to 0.5344 and recall to 0.6863, while soft voting reached F1 of 0.5507 and recall of 0.7451, pulled down by KNN's poor recall (0.3333) and F1 (0.3696).
- R4.3: After round 1 normalisation, the two misclassified points (Points 3 and 7) each carry weight 0.2500, totalling 0.5000 (50% of the distribution). Round 2 assigns a higher voting weight to h2 (α2 = 0.8047 vs α1 = 0.5493) because h2 achieved a lower reweighted error of 0.1667 compared to round 1's error of 0.2500.
- R4.4:
  - Model A Option 2 (CNN) layer breakdown:
  - Input: output shape 28 × 28 × 1, 0 parameters
  - Conv1 (5×5, pad 2, stride 1): output shape 28 × 28 × 6, 156 parameters
  - MaxPool1 (2×2, stride 2): output shape 14 × 14 × 6, 0 parameters
  - Conv2 (5×5, pad 0, stride 1): output shape 10 × 10 × 16, 2,416 parameters
  - MaxPool2 (2×2, stride 2): output shape 5 × 5 × 16, 0 parameters
  - Flatten: output shape 400, 0 parameters
  - Dense1 (ReLU): output shape 32, 12,832 parameters
  - Dense2 (Output): output shape 10, 330 parameters
  - Total CNN parameters: 15,734
  - For Model B, weight matrix shapes are W_xh: (128, 64), W_hh: (128, 128), and W_hy: (16, 128). rnn_total does not scale with sequence length because the recurrent weights and biases are shared across all sequential steps.
  - For 10-class routing, use ReLU hidden activations, a Softmax output layer, and Categorical Cross-Entropy loss.
  - In very deep plain networks, repeated backpropagation multiplications cause gradients to diminish exponentially, stalling parameter updates in early layers. Skip connections introduce an identity bypass path, enabling gradient flow directly to early layers without attenuation.

>>>>>>>## Model-choice memo
To: Head of Scheduling Operations, PharmEasy  
From: Operations Analytics Team  
Subject: Production Recommendation for Appointment No-Show Alerts  

recommend deploying the regularized Logistic Regression model at an alert threshold of τ = 0.1500 for live operations. 

On test evaluation, this model captures 94.12% of actual no-shows (recall 0.9412, catching 48 of 51 missed appointments). Across the test cohort, operational costs total INR 6,050, delivering a 48.7% cost reduction compared to the standard τ = 0.5000 cutoff (which costs INR 11,800 with only 47.06% recall) and saving INR 14,350 relative to the unmanaged baseline cost of INR 20,400.

At τ = 0.3000, logistic regression achieved an F1 score of 0.5714 and recall of 0.7451, decisively beating standalone KNN (F1 0.3696, recall 0.3333), decision stumps (F1 0.5532), and voting ensembles (hard voting F1 0.5344; soft voting F1 0.5507). Ensemble voting was dragged down by KNN's weak neighborhood density and offers no operational advantage. Tuning τ to 0.1500 aligns directly with our unit economics: an unflagged empty slot costs PharmEasy roughly INR 400 in lost provider time, whereas an unnecessary outbound reminder call costs merely INR 50.

Operational Risk: threshold was tuned on 560 training records with synthetic behavioral assumptions. Shifts in outbound call center costs or patient call fatigue could alter the optimal operating point.

Future Architectures: Neither Model A (scan routing CNN) nor Model B (referral text RNN) should be prioritized before deploying this structured scheduling model, as scheduling triage generates immediate cash savings on proven tabular pipelines without requiring unvalidated deep learning infrastructure.

>>>>## LIMITATIONS

1. Synthetic Data Realism: All patient records were generated through mathematical sampling equations with fixed parametric assumptions. True healthcare behaviors involve complex systemic interactions, clinic operational delays, transport barriers, and seasonal morbidities not captured in this synthetic set.
2. Single Deterministic Train/Test Split: Analysis relies on a single 70/30 stratified split (560 training, 240 test records). While deterministic splitting ensures exact experimental reproducibility, it leaves reported threshold and test costs subject to split-specific variance that repeated k-fold nested cross-validation would better quantify.
3. Low Positive Test Class Support: The test set contains only 51 positive no-show events. With small positive support, single-patient misclassifications swing precision and recall rates by approximately two full percentage points, introducing volatility to sample metric estimates.
