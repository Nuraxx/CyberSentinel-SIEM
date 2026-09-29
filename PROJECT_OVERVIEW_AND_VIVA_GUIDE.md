# CyberThreat-ML — Project Overview & Viva Guide
### Aligned to the official 23CSE301 Machine Learning Capstone Guidelines (AY 2026-27)

*Written for our 3-person team. Every fact here comes from actually inspecting this repository and running its notebooks — nothing is invented. Wherever the repo doesn't explicitly answer something, it says "Not verified" instead of guessing. Sections A/B/C below separate what kind of claim each part is making, per the official guideline on originality:*

- **[VERIFIED FACT]** — confirmed by reading the code/notebooks or running them this session.
- **[ML CONCEPT]** — a general explanation of how an algorithm/metric works, not specific to our data.
- **[TEAM TO CONFIRM]** — an interpretation or observation that reads the data honestly, but which the team must personally review, understand, and be ready to defend as their own in viva — per the guideline that analysis and interpretation must be original, not just AI-generated text repeated back.

---

## 1. Project Introduction

### A. 30-second introduction
"Our project, CyberThreat-ML, uses the CICIDS2017 network-traffic dataset to build three ML pipelines: a **regression** model that predicts a 0–100 cybersecurity risk score, a **classification** model that identifies the attack type, and **clustering** that groups traffic by behavior without using labels. All three tracks share one cleaned dataset and follow the same fair-comparison rules — same train/test split, same random seed, real evaluation metrics."

### B. 1–2 minute introduction
"We're using the CICIDS2017 dataset — a real, published cybersecurity research dataset where the Canadian Institute for Cybersecurity captured 5 days of network traffic and deliberately ran attacks (DoS, DDoS, brute-force, port scans, web attacks, botnets) alongside normal traffic, then summarized every connection into flow-level statistics.

The problem: security analysts can't manually inspect every connection on a busy network, and signature-based detection struggles against attacks that don't exactly match a known pattern. We explore whether classical machine learning, trained directly on flow statistics, can help.

Our project covers all three required tracks. **[VERIFIED FACT]** **Regression** predicts a custom-built 0–100 risk score we designed ourselves from behavioral indicators — not a label lookup. **Classification** predicts which of 15 categories (14 attack types + BENIGN) a flow belongs to. **Clustering** groups flows by behavioral similarity with no labels used during fitting, to see if the traffic naturally separates the way we'd expect.

The overall objective is an honest, leakage-free, fairly-compared pipeline — not necessarily the single highest score, but one where every score can be trusted because the same rules were applied everywhere. The useful outcome is a demonstration that flow-level features alone carry enough signal to support triage-style security decisions."

### C. 3–5 minute detailed introduction
Combine B above, then add:

"Structurally, the project has 4 notebooks. `01_EDA.ipynb` is not one of the three graded tracks itself — it's shared groundwork: it loads and merges the 8 raw CICIDS2017 CSVs, audits and cleans the data once, explores it, and engineers one new feature, so the regression, classification, and clustering notebooks all build on exactly the same, already-cleaned dataset instead of each doing their own (potentially inconsistent) cleaning.

**[VERIFIED FACT]** For regression, all 10 officially required algorithms are trained and compared on the same held-out test split, with hyperparameter tuning and 5-fold cross-validation on the top 2. For classification, all 10 officially required algorithms (5 in Part A: Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVM; 5 in Part B: Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP) are trained, with a single consolidated comparison table, tuning, and a SMOTE-based imbalance comparison. For clustering, K-Means and Agglomerative Hierarchical Clustering are both fitted with labels withheld, evaluated with Silhouette/Davies-Bouldin/Calinski-Harabasz, and interpreted against real labels only afterward.

**[TEAM TO CONFIRM]** Our headline finding is that this data has a severe, ~190,000:1 class imbalance and a real leakage risk in the `destination_port` column (most attack types sit on one fixed port) — both of which we found ourselves by inspecting the actual data, and both of which shaped concrete design decisions (class weighting, SMOTE, macro-F1 reporting, and an explicit flag to check feature importance against the port finding)."

---

## 2. Official Assignment Structure (from the attached PDF)

**[VERIFIED FACT — from the official PDF, not our repo]**

| Item | Official requirement |
|---|---|
| Team size | 3 members |
| Tracks | Regression, Classification, Clustering |
| Reviews | Two formal, in-person reviews, 25 marks each (50 total) |
| Datasets | Assigned by instructor, one per team |
| Submission | Jupyter Notebook(s) + GitHub repository |

**Review 1** (just before mid-semester exams) — scope: **Full Regression track + Classification Part A**. 25 marks: Dataset & EDA (4), Preprocessing & Feature Engineering (3), Regression Track (9), Classification Part A (3), Presentation (1), Viva (5).

**Review 2** (towards end of semester) — scope: **Classification Part B + Full Clustering track**. 25 marks: Classification Part B (6), Clustering Track (8), Pipeline Integration & Quality (3), Presentation & Viva (8).

Both reviews are in-person; **any team member can be questioned about any part of the work**, not just their own track.

**Bonus (Review 2 only, up to +2, official rubric caps the displayed total at 20 for grading unless the instructor says otherwise):**
- +1 — a working GUI (Streamlit/Gradio) that accepts input and returns predictions.
- +1 — that GUI publicly deployed (e.g. Streamlit Community Cloud, Hugging Face Spaces, Render).
- +2 — both.

---

## 3. Exact Official Algorithm List — What We Actually Implemented, and Results

### Regression (all 10 required — Review 1)

| # | Algorithm | Basic idea [ML CONCEPT] | Why used | Key params tuned/used | Our implementation & actual result [VERIFIED FACT] |
|---|---|---|---|---|---|
| 1 | Linear Regression | Fits one straight-line relationship between features and target. | Baseline; coefficients are directly interpretable. | (none) | Trained on all 79 features. R² = 0.7235, RMSE = 7.090, MAE = 4.486. |
| 2 | Ridge Regression | Linear regression + L2 penalty shrinking coefficients. | Handles the 71 correlated feature pairs found in EDA more gently than plain linear regression. | `alpha=1.0` | R² = 0.5518, RMSE = 9.027, MAE = 4.525 (ranked *lower* than plain Linear Regression in our data — a real, reported result). |
| 3 | Lasso Regression | Linear regression + L1 penalty; can zero out coefficients entirely. | Can automatically discard uninformative features. | `alpha=0.1` | R² = 0.6782, RMSE = 7.649, MAE = 5.837. |
| 4 | ElasticNet Regression | Blend of L1 + L2 penalties. | Balances Ridge and Lasso's effects. | `alpha=0.1`, `l1_ratio=0.5` | R² = 0.6744, RMSE = 7.694, MAE = 5.904. |
| 5 | Polynomial Regression | Expands features (e.g. squared, interaction terms) then fits linear regression. | Captures curved relationships. | `degree=2` | **Important implementation note:** applied only to the 5 raw risk-score indicators (not all 79 features), since expanding 79 features to degree 2 would generate an impractically large number of terms (21 terms from 5 inputs). R² = 0.0182, RMSE = 13.361, MAE = 11.879 — lowest of all 10, expected given the restricted feature set, not a modeling failure. |
| 6 | Decision Tree Regressor | Learns if/else splits, predicts the average value per resulting region. | Interpretable; shows feature importance directly. | `max_depth=15` (tuned further, see Part 10) | R² = 0.9985, RMSE = 0.517, MAE = 0.170. |
| 7 | Random Forest Regressor | Many decision trees on random subsets, averaged. | Ensemble baseline; became the overall best model. | `n_estimators=150`, `max_depth=15` (tuned further) | R² = 0.9992, RMSE = 0.392, MAE = 0.139 — **rank #1**. |
| 8 | Gradient Boosting Regressor | Sequential trees, each fit to the residual errors of the previous ones. | Strong ensemble comparison point. | `n_estimators=100`, `max_depth=3` (`sklearn` GBM, not XGBoost) | R² = 0.9939, RMSE = 1.049, MAE = 0.681. |
| 9 | Support Vector Regressor (SVR) | Fits predictions within an error-tolerance "tube," penalizing points outside it. | Different, margin-based comparison point. | `kernel="rbf"` (features scaled first, as the algorithm list requires) | R² = 0.8854, RMSE = 4.565, MAE = 2.385. **Implementation note (a real bug we found and fixed):** SVR is trained on a smaller stratified subsample of the training split (20,000 rows) for runtime reasons, but is evaluated on the *same shared held-out test split* as the other 9 models. |
| 10 | K-Nearest Neighbors Regressor | Predicts the average target value of the nearest training points. | Simple, non-parametric baseline. | `n_neighbors=5` (features scaled first) | R² = 0.9877, RMSE = 1.494, MAE = 0.455. |

**Evaluation metrics — all mandatory per the PDF, all present [VERIFIED FACT]:** R², RMSE, MAE for all 10 in one ranked table; 5-fold cross-validated R² for the top 2 (Random Forest: 0.9992 ± 0.0001; Decision Tree: 0.9985 ± 0.0001).

### Classification Part A (Review 1 — the 5 required)

| # | Algorithm | Basic idea [ML CONCEPT] | Why used | Key params | Our result [VERIFIED FACT] |
|---|---|---|---|---|---|
| 1 | Logistic Regression | Linear probability boundary between classes (softmax for multiclass). | Baseline; coefficients/odds are interpretable. | `class_weight="balanced"`, `max_iter=1000` | Accuracy 0.9480, F1 weighted 0.9542. |
| 2 | K-Nearest Neighbors | Majority vote among the k closest training points. | Simple distance-based comparison. | `n_neighbors=5` | Accuracy 0.9802, F1 weighted 0.9798. |
| 3 | Gaussian Naive Bayes | Assumes features are independent & normally distributed per class. | Very fast, tests the independence assumption directly. | (none tuned) | Accuracy 0.8898, F1 weighted 0.8987. |
| 4 | Decision Tree Classifier | If/else splits on feature thresholds. | Interpretable; visualizable. | `max_depth=20`, `class_weight="balanced"` | Accuracy 0.9859, F1 weighted 0.9863. |
| 5 | Support Vector Machine (SVC) | Widest-margin boundary between classes (RBF kernel here). | Strong on complex, non-linear boundaries. | `C`, `gamma` (features scaled first) | Accuracy 0.9433, F1 weighted 0.9451. **Implementation note (same fix as SVR above):** trained on a stratified 20,000-row subsample of the training split for runtime, evaluated on the *same shared test split* as the other 9 classifiers. |

**Preliminary comparison table (Part A, all 5), ranked:** Decision Tree (0.9863) > KNN (0.9798) > Logistic Regression (0.9542) > SVM (0.9451) > Gaussian Naive Bayes (0.8987), by F1 weighted.

### Classification Part B (Review 2 — the remaining 5)

| # | Algorithm | Basic idea [ML CONCEPT] | Why used | Key params | Our result [VERIFIED FACT] |
|---|---|---|---|---|---|
| 6 | Random Forest Classifier | Many decision trees on random subsets, majority vote. | Ensemble robustness; also used for feature importance. | `n_estimators=150`, `max_depth=20`, `class_weight="balanced"` (tuned further, see Part 10) | Accuracy 0.9851, F1 weighted 0.9856. |
| 7 | AdaBoost Classifier | Sequential weak learners, each focusing on the previous ones' mistakes. | Classic boosting comparison. | `n_estimators=100` | Accuracy 0.6210, F1 weighted 0.5857, **F1 macro only 0.449** — by far our weakest model, and its macro score shows it specifically fails on rare classes. |
| 8 | Gradient Boosting Classifier | Sequential trees fit to residual errors (`sklearn` GBM). | Strong ensemble comparison. | `n_estimators=100`, `max_depth=3` | Accuracy 0.9762, F1 weighted 0.9774. |
| 9 | Bagging Classifier | Many base estimators (Decision Trees, per the PDF's note) on bootstrap samples, averaged. | Ended up our **overall best model**. | `n_estimators=50` (Decision Tree base estimator, scikit-learn's default) | Accuracy 0.9875, F1 weighted 0.9873 — **rank #1 of all 10**. |
| 10 | MLP Classifier (Neural Network) | Layers of weighted connections, learned via backpropagation. | Captures complex non-linear patterns. | `hidden_layer_sizes=(100,50)` (tuned further, see Part 10) | Accuracy 0.9832, F1 weighted 0.9800. |

**Full 10-algorithm comparison table (Review 2 requirement), ranked by F1 weighted [VERIFIED FACT]:**

| Rank | Model | Accuracy | Precision (wtd) | Recall (wtd) | F1 (wtd) | F1 (macro) | ROC-AUC (OvR) |
|---|---|---|---|---|---|---|---|
| 1 | Bagging | 0.9875 | 0.9872 | 0.9875 | 0.9873 | 0.9101 | 0.9992 |
| 2 | Decision Tree | 0.9859 | 0.9876 | 0.9859 | 0.9863 | 0.9312 | 0.9956 |
| 3 | Random Forest | 0.9851 | 0.9875 | 0.9851 | 0.9856 | 0.8924 | 0.9998 |
| 4 | MLP | 0.9832 | 0.9858 | 0.9832 | 0.9800 | 0.8919 | 0.9997 |
| 5 | KNN | 0.9802 | 0.9798 | 0.9802 | 0.9798 | 0.8795 | 0.9971 |
| 6 | Gradient Boosting | 0.9762 | 0.9818 | 0.9762 | 0.9774 | 0.7940 | 0.9877 |
| 7 | Logistic Regression | 0.9480 | 0.9673 | 0.9480 | 0.9542 | 0.7997 | 0.9975 |
| 8 | SVM | 0.9433 | 0.9642 | 0.9433 | 0.9451 | 0.7708 | 0.9976 |
| 9 | Gaussian Naive Bayes | 0.8898 | 0.9254 | 0.8898 | 0.8987 | 0.7398 | 0.9923 |
| 10 | AdaBoost | 0.6210 | 0.7236 | 0.6210 | 0.5857 | 0.4487 | 0.9453 |

**On ROC-AUC and One-vs-Rest [ML CONCEPT]:** ROC-AUC normally applies to a single yes/no decision. With 15 classes, we use **One-vs-Rest (OvR)**: for each class, treat it as "this class" vs. "everything else," compute that class's ROC-AUC, then average — this is why `utils.evaluate_classifier()` uses `roc_auc_score(..., multi_class="ovr")`.

---

## 4. Dataset Overview

**[VERIFIED FACT]**

| Item | Value |
|---|---|
| Dataset name | CICIDS2017 (Intrusion Detection Evaluation Dataset, 2017) |
| Source | Canadian Institute for Cybersecurity, University of New Brunswick — cited in `README.md` with the official dataset URL |
| Why selected | Per the README, chosen to explore whether flow-level behavioral features can support detection, risk-scoring, and behavioral clustering — a real, published dataset with genuine attacks captured alongside normal traffic |
| Rows before cleaning | 2,830,743 (merged from 8 raw day-files) |
| Rows after cleaning | 2,520,798 |
| Features (raw) | 78 numeric + 1 target (`label`) = 79 columns |
| Features (after our feature engineering) | 79 numeric + `label` = 80 columns |
| Target column | `label` |
| What one row represents | One network flow — a summarized connection, not a single packet |
| Important columns | `flow_duration`, `flow_bytess`/`flow_packetss` (rates), `total_fwd_packets`/`total_backward_packets`, TCP flag counts, `destination_port` |
| Numerical vs. categorical | All 78 raw feature columns are numeric; `label` is the only text column; there are **no other categorical columns** to encode |
| Target classes | 15: `BENIGN`, `Bot`, `DDoS`, `DoS GoldenEye`, `DoS Hulk`, `DoS Slowhttptest`, `DoS slowloris`, `FTP-Patator`, `Heartbleed`, `Infiltration`, `PortScan`, `SSH-Patator`, `Web Attack - Brute Force`, `Web Attack - Sql Injection`, `Web Attack - XSS` |
| Class imbalance | 190,459.7 : 1 (`BENIGN` = 2,095,057 rows / 83.11%, vs. `Heartbleed` = 11 rows) |
| Missing values | No raw `NaN`s, but 2,867 rows (0.10%) had `Infinity` (division by a zero flow duration) — converted to `NaN`, dropped |
| Duplicates | 307,078 exact duplicate rows (10.85%) removed |
| Outliers | Checked via IQR across all 78 columns; worst columns flag 20–24% of rows; **kept, not removed** (justification in Part 8) |
| Data-quality issues found | (1) A duplicate-looking column pair, `fwd_header_length`/`fwd_header_length1`, with identical values — a quirk of the source tool. (2) Garbled text (`Web Attack ï¿½ XSS`) in 3 label values from a text-encoding mismatch — cosmetic only, doesn't affect modeling. (3) `flow_duration` has a minimum of **-13** (negative), a known CICIDS2017 data-quality artifact we documented but did not attempt to "fix." |

**Columns confirmed NOT present** (checked directly against the cleaned CSV, so this is not a guess): `Source IP`, `Destination IP`, `Source Port`, a `Protocol` (TCP/UDP) column, and `Timestamp` do not exist in this dataset variant. Only `destination_port` survives as an identifier-like field.

---

## 5. Cybersecurity Background (only concepts actually present)

**Network traffic / Network flow** — *Simple:* all the "conversations" on a network; a flow is one such conversation. *Technical:* CICFlowMeter groups packets sharing endpoints into one aggregated record. *In our data:* every row **is** one flow.

**Packets** — *Simple:* the small chunks of data making up a flow. *In our data:* never seen individually, only as aggregate counts (`total_fwd_packets`, `total_backward_packets`).

**Destination port** — *Simple:* the number telling a receiver which service traffic is meant for (80 = web, 21 = file transfer, 22 = secure login). *Technical:* a 16-bit transport-layer field. *In our data:* the one identifier-like column kept; **[TEAM TO CONFIRM]** our own check found 11 of 15 classes sit 100% on one port — a real leakage risk we chose to document rather than silently exploit or silently drop.

**TCP flags (SYN, ACK, FIN, RST, PSH, URG, CWE, ECE)** — *Simple:* signal bits marking a connection's state. *In our data:* used as raw features, and combined into our custom risk score's "flag anomaly" indicator.

**DoS / DDoS** — *Simple:* flooding a target (from one machine, or many at once) so it can't serve real users. *In our data:* `DDoS`, `DoS Hulk`, `DoS GoldenEye`, plus the "low and slow" variants `DoS Slowhttptest`/`DoS slowloris` (holding connections open with minimal data instead of flooding).

**Brute force** — *Simple:* trying many credentials until one works. *In our data:* `FTP-Patator`, `SSH-Patator`, `Web Attack - Brute Force`.

**Botnet** — *Simple:* a network of compromised machines controlled remotely. *In our data:* the `Bot` class; **[TEAM TO CONFIRM]** our scatter-plot analysis found Bot traffic forms a visually distinct cluster on a duration-vs-packet-rate plot, consistent with regular C2 "beacon" behavior.

**Port scanning** — *Simple:* probing many ports to find open services. *In our data:* the `PortScan` class — and notably, the *only* class our leakage check found spread across many ports (0.4% concentration) rather than one, which is exactly what scanning should look like.

**Web attacks (Brute Force, XSS, SQL Injection)** — attacks against a website itself. *In our data:* three separate, very rare classes.

**Infiltration** — *Simple:* an attacker already inside the network. *In our data:* a class with only 36 rows.

**Heartbleed** — *Simple:* a specific 2014 bug (CVE-2014-0160) in OpenSSL that let attackers read server memory. *In our data:* the rarest class (11 rows).

*Not covered: Source/Destination IP and a distinct Protocol/TCP-UDP marker are not present as columns in this dataset (verified in Part 4), so they're not discussed as project-specific concepts.*

---

## 6. Complete Project Pipeline

```
Raw CICIDS2017 CSVs (8 files, data/raw/)
  ↓ Data loading      merge all 8 files into one table
  ↓ Data audit        shape, dtypes, missing-value counts, class distribution
  ↓ EDA               distributions, correlation heatmap, scatter plots, target plot
  ↓ Cleaning          Infinity → NaN → drop; drop exact duplicates
  ↓ Feature eng.       add header_payload_ratio
  ↓ Encoding          label-encode the target (no categorical features exist)
  ↓ Scaling           StandardScaler, fit on train split only
  ↓ Train/test split  80:20, stratified, random_state=42, one split per track
  ↓
  ├─ Regression (10 algorithms) → R²/RMSE/MAE table → tuning (top 2) → residual/actual-vs-predicted/importance plots
  ├─ Classification (10 algorithms) → Accuracy/Precision/Recall/F1/ROC-AUC table → tuning (top 3) + SMOTE → confusion matrices
  └─ Clustering (2 algorithms, labels withheld) → Silhouette/DB/CH → PCA & t-SNE plots → labels used only now, to interpret
  ↓
Final results & written conclusions (per notebook)
```

For what happens/why/output at each stage, see Part 6 of the pipeline table in Section 1 above and the notebook-by-notebook breakdown in Part 7 below — repeating it a third time would be redundant.

---

## 7. Notebook Overview

All four notebooks exist and, as of this session, **execute top-to-bottom from a clean kernel with zero errors** (verified by parsing every cell's output for error-type results, not just assumed from the code existing).

| Notebook | Purpose | Input | Models | Metrics | Output |
|---|---|---|---|---|---|
| `01_EDA.ipynb` | Load, audit, clean, explore, enrich the raw data once | 8 raw CSVs | None (no modeling) | N/A | `data/processed/cleaned_dataset.csv` |
| `02_Classification.ipynb` | Predict attack type | cleaned CSV | 10 classifiers (Part A + Part B) | Accuracy, Precision/Recall (wtd), F1 (wtd+macro), ROC-AUC (OvR), confusion matrix | Saved model + scaler + encoder in `models/classification/` |
| `03_Regression.ipynb` | Predict 0–100 risk score | cleaned CSV | 10 regressors | R², RMSE, MAE | Saved model + risk-score calculator + scaler in `models/regression/` |
| `04_Clustering.ipynb` | Discover behavioral groups, unsupervised | cleaned CSV | K-Means, Agglomerative | Silhouette, Davies-Bouldin, Calinski-Harabasz | Saved K-Means model + PCA transformer + cluster profile in `models/clustering/` |

*Why `01_EDA.ipynb` exists as a 4th notebook, given the PDF's structure lists only 3 tracks:* EDA/cleaning/feature-engineering is graded (Section A/B of Review 1) but isn't itself one of the three modeling **tracks** — it's shared groundwork feeding all three. The PDF's guideline 7.1 explicitly allows "one notebook per track... **or** a single notebook with clearly labelled sections," and the example filenames given (`regression.ipynb`, `classification.ipynb`, `clustering.ipynb`) are introduced with "e.g." — read as illustrative, not a literal mandatory filename list. **[TEAM TO CONFIRM WITH INSTRUCTOR]** this structure (one shared EDA/preprocessing notebook + one notebook per track) is a reasonable reading of that rule, but it's worth explicitly confirming acceptable with the instructor before Review 1, since it's a structural choice, not something the PDF spells out in this exact shape.

---

## 8. EDA + Preprocessing (detail)

**[VERIFIED FACT]** — `01_EDA.ipynb`, in order:
- **Shape/dtypes:** `df.shape` and `df.info()` → 2,830,743 × 79; 78 numeric, 1 text.
- **Missing values:** `clean_data()` reports 2,867 rows (0.10%) with Infinity, converted to NaN, dropped.
- **Duplicates:** 307,078 rows (10.85%) removed.
- **Class/target distribution:** a table + bar chart (linear and log scale) of all 15 classes.
- **Feature distributions:** a 5-panel `log1p` view of key features, plus a 79-panel raw-scale grid covering *every* numeric feature (satisfying the PDF's "distribution plots for each feature").
- **Correlation heatmap:** full 78×78 heatmap + a table of the 71 pairs with |r| ≥ 0.9.
- **Target plot:** the class-distribution bar chart above.
- **Scatter plots (≥2, feature-target relationships, per the PDF):** duration-vs-packet-rate and byte-rate-vs-engineered-ratio, both colored by `label`.
- **Written observations:** a Markdown cell after every major plot, grounded in the real numbers above (not generic text) — satisfying the "insight commentary" requirement.
- **Outliers:** IQR check across all 78 numeric columns, 20–24% flagged on the worst columns, kept (not removed) with a written justification.
- **Encoding:** only the target (`LabelEncoder`) — no categorical input features exist.
- **Scaling:** `StandardScaler`, fit on the training split only, in every track's notebook.
- **Train/test split:** 80:20, `random_state=42`, `stratify=True` — same split reused by every model within a track.
- **Stratification:** confirmed via `train_test_split(..., stratify=...)` in `src/preprocessing.py`'s `split_data()`.
- **Data leakage — how it's avoided:** the scaler and the custom risk score's percentile boundaries are both fit on the training split only, never on the full dataset before splitting.

---

## 9. Feature Engineering

**[VERIFIED FACT]** — exactly one engineered feature exists in the project:

- **Name:** `header_payload_ratio`
- **Original columns used:** `fwd_header_length`, `bwd_header_length` (header bytes); `total_length_of_fwd_packets`, `total_length_of_bwd_packets` (payload bytes)
- **Formula:** `(fwd_header_length + bwd_header_length) / (total_length_of_fwd_packets + total_length_of_bwd_packets + 1)`
- **Meaning:** how much of a flow's bytes are protocol overhead vs. actual data.
- **Why created:** none of the 78 raw columns expresses this relationship directly.
- **Potential benefit:** legitimate bulk-data flows have large payloads relative to fixed header overhead (small ratio); probing/incomplete-connection traffic is mostly header (large ratio) — a cheap, interpretable, pre-computed signal instead of forcing a model to learn the division implicitly.
- **[TEAM TO CONFIRM] Actual evidence from our project:** in a 6,000-row stratified scatter sample, the single most extreme value of this ratio belongs to a real `DoS Slowhttptest` flow — an attack that specifically works by sending almost no payload while holding a connection open. This is a genuine pattern found in the data, not an assumed one, but the team should look at the actual scatter plot (`reports/figures/scatter_byterate_vs_headerratio.png`) themselves before presenting this as their own finding in viva.

No second engineered feature currently exists — do not claim otherwise in viva.

---

## 10. Regression Track — Review 1 (full detail)

**Target:** a custom 0–100 **behavior-based risk score**, computed by `src/risk_score.py`, from 5 indicators (packet rate, byte rate, TCP-flag anomaly, duration extremity, forward/backward traffic asymmetry), each percentile-normalized **on training data only**, combined with fixed weights (25/20/25/15/15%). **Not** a `BENIGN=0, DDoS=100` lookup table.

**Why regression:** "how risky is this?" is naturally a continuous question; regression lets us build a designed, explainable score rather than only depending on the fixed attack label.

**Preprocessing:** same cleaned dataset as every other track; split happens *before* the risk score is generated, so the score's own training statistics never see test rows; `StandardScaler` fit on train only.

**Train/test split:** 80:20, stratified, `random_state=42` — **the same preprocessed dataset and same held-out test set for all 10 models**, per the PDF's explicit requirement — with the one documented exception that SVR trains on a smaller *training-only* subsample for speed while still being *evaluated* on the identical shared test set as the other 9.

**Model training & predictions:** all 10 models are `.fit()` and `.predict()` in `03_Regression.ipynb`, with per-model timing printed, zero execution errors (verified this session via a clean-kernel run).

**Comparison table, ranked by R² [VERIFIED FACT]:**

| Rank | Model | R² | RMSE | MAE |
|---|---|---|---|---|
| 1 | Random Forest | 0.9992 | 0.392 | 0.139 |
| 2 | Decision Tree | 0.9985 | 0.517 | 0.170 |
| 3 | Gradient Boosting | 0.9939 | 1.049 | 0.681 |
| 4 | KNN Regression | 0.9877 | 1.494 | 0.455 |
| 5 | SVR | 0.8854 | 4.565 | 2.385 |
| 6 | Linear Regression | 0.7235 | 7.090 | 4.486 |
| 7 | Lasso | 0.6782 | 7.649 | 5.837 |
| 8 | ElasticNet | 0.6744 | 7.694 | 5.904 |
| 9 | Ridge | 0.5518 | 9.027 | 4.525 |
| 10 | Polynomial Regression | 0.0182 | 13.361 | 11.879 |

**5-fold cross-validated R² for the two best-performing models (mandatory):** Random Forest 0.9992 ± 0.0001; Decision Tree 0.9985 ± 0.0001.

**Hyperparameter tuning (`RandomizedSearchCV`, cv=3, on the top 2):**

| Model | Best params found | R² before→after | RMSE before→after | MAE before→after |
|---|---|---|---|---|
| Random Forest | `n_estimators=150, min_samples_split=2, max_features='sqrt', max_depth=None` | 0.9992 → **0.9993** | 0.392 → **0.348** | 0.139 → **0.133** |
| Decision Tree | `min_samples_split=5, min_samples_leaf=2, max_depth=30` | 0.9985 → 0.9985 (flat) | 0.517 → 0.521 (slightly worse) | 0.170 → **0.142** (better) |

**[TEAM TO CONFIRM]** we report Decision Tree's mixed result honestly (not every metric improves after tuning every time) rather than only showing favorable numbers.

**Visualizations (mandatory, for the best model, Random Forest):** actual-vs-predicted scatter plot, residual plots (residuals-vs-predicted + residual histogram), and a tree-based feature-importance bar chart — all titled, axis-labelled, `tight_layout`-formatted.

---

## 11. Classification — Review 1, Part A (full detail)

Covered algorithm-by-algorithm with real results in Part 3 above. Summary:

- **Problem:** predict which of 15 `label` categories a flow belongs to.
- **Target:** `label`, label-encoded to integers 0–14.
- **Preprocessing:** stratified 150,000-row working sample (73,901 rows in practice, since most classes are smaller than the per-class cap); 80:20 stratified split; `StandardScaler` fit on train only.
- **Split:** same shared split used by all 5 Part-A algorithms (and, after our fix, by SVM specifically evaluated on that same shared test split rather than a separate one).
- **Training/predictions:** all 5 trained and predict with zero errors.
- **Accuracy / Weighted F1 / Confusion Matrix:** reported per algorithm (see Part 3 table); confusion matrices shown both as a compact grid across all 10 models and in full labelled detail for the best model so far.
- **Preliminary comparison table (Part A only):** Decision Tree (F1 wtd 0.9863) > KNN (0.9798) > Logistic Regression (0.9542) > SVM (0.9451) > Gaussian Naive Bayes (0.8987).

---

## 12. Classification — Review 2, Part B (full detail)

Covered algorithm-by-algorithm with real results in Part 3 above. Summary:

- **Final 10-model comparison table:** see Part 3 — includes Accuracy, Precision (weighted), Recall (weighted), F1 (weighted), F1 (macro, extra beyond the mandatory list), and ROC-AUC (OvR), exactly the metric set the PDF requires for Review 2 (plus one bonus metric).
- **Tuning:** `RandomizedSearchCV` (cv=3) on Random Forest, SVM, and MLP:

| Model | F1 (weighted) before → after |
|---|---|
| SVM | 0.9451 → **0.9697** (clear improvement) |
| Random Forest | 0.9856 → **0.9869** (small improvement) |
| MLP | 0.9800 → 0.9779 (**got slightly worse** — reported honestly) |

- **Model-selection process:** after tuning, a focused SMOTE comparison is applied to the single strongest tuned candidate (Random Forest, tuned) — F1 weighted stayed essentially flat (0.9869 → 0.9869) but F1 macro reached 0.9252, which is the number that actually reflects rare-class improvement. All 14 candidates (10 baselines + 3 tuned + 1 SMOTE variant) are then compared, and **Bagging remains the overall best** (F1 weighted 0.9873) — notably, Bagging was never itself one of the 3 models chosen for tuning, since it was already the strongest baseline.
- **Confusion matrices:** a grid of all 10 (row-normalized) plus a fully labelled one + ROC curves for the best model.

---

## 13. Clustering — Review 2 (full detail)

- **Why clustering:** to check for behavioral groupings the fixed 15-label scheme might not fully capture, with no ground truth used during fitting.
- **Why labels aren't used during fitting:** `true_labels_kmeans`/`true_labels_hier` are set aside immediately after loading and never referenced again until the interpretation section at the end — verified directly in the code.
- **Preprocessing:** 33 highly-correlated features (|r| > 0.9) dropped before scaling (distance-based methods are more sensitive to redundant features than trees/regularized regression); `StandardScaler` fit once and reused; PCA reduces to 22 components (95% variance retained) for the actual clustering, plus a separate 2-component PCA purely for visualization.
- **Number of clusters:** k = 10 for K-Means, chosen via the silhouette score across k = 2–10 (the mandatory elbow curve is also shown).
- **Algorithm logic:** K-Means iteratively assigns points to the nearest of k centers; Agglomerative Clustering repeatedly merges the closest pair of clusters, visualized via a dendrogram.
- **A specific, real, presentable finding:** comparing linkage strategies, "average" linkage scored a *higher* raw silhouette (0.69) than "ward" (0.45) — but was rejected because it dumped 99.1% of points into a single cluster (a degenerate, useless split); "ward" (46.2% largest cluster) was chosen as the genuinely balanced option instead.
- **Evaluation (all 3 mandatory metrics, for both algorithms):**

| Metric | K-Means | Agglomerative (ward) |
|---|---|---|
| Silhouette | 0.426 | 0.453 |
| Davies-Bouldin | 0.899 | 1.021 |
| Calinski-Harabasz | 3709.8 | 1090.6 |

  (not directly comparable to each other — different sample sizes, as documented in the notebook's own "possible mistakes" section)

- **Visualizations (all mandatory per the PDF):** Elbow curve (K-Means) ✓, Dendrogram (Agglomerative) ✓, PCA 2D scatter **for both algorithms** ✓ (`kmeans_pca_2d.png`, `agglomerative_pca_2d.png`), t-SNE for at least one algorithm ✓ (Agglomerative).
- **[TEAM TO CONFIRM] Interpretation (labels brought back only now):** several clusters are highly "pure" — e.g. one cluster is 98.9% `DoS GoldenEye`, another 98.8% `DoS Slowhttptest`, another 90.9% `Heartbleed` — while the single largest cluster (15,240 points) is a mixed bag dominated by `PortScan` at only 17.1%, showing genuine behavioral overlap rather than a forced 1-to-1 mapping to the 15 labels. The team should look at `cluster_profile` in the notebook themselves and be ready to describe this in their own words.

---

## 14. Pipeline Integration & Quality (Review 2, Section C)

**[VERIFIED FACT]**
- **Notebooks run top-to-bottom without errors:** confirmed this session via a clean-kernel `jupyter nbconvert --execute` on all 4 notebooks — 0 errors across 157 total cells.
- **Modular functions:** shared logic lives in `src/preprocessing.py` (`load_and_merge_csvs`, `standardize_columns`, `clean_data`, `check_outliers`, `engineer_features`, `encode_labels`, `split_data`, `scale_features`, `stratified_sample`), `src/risk_score.py` (`RiskScoreCalculator`), and `src/utils.py` (plotting/evaluation helpers) — not repeated inline per notebook.
- **Markdown/comments:** every major code block in every notebook is preceded by a Markdown heading/explanation; no unexplained "wall of code" sections found.
- **`random_state=42`:** used consistently across `src/` and all 4 notebooks (verified by direct search).
- **No data leakage:** scalers, encoders, and the custom risk score are all fit on training data only, confirmed by reading the code, not assumed.
- **Consistent train/test split:** confirmed for every model within each track, including the SVM/SVR fix described in Part 15 below.
- **Summary tables instead of scattered prints:** both tracks' comparisons use a single `pandas.DataFrame`, not individual print statements.

---

## 15. README + GitHub Requirements — Checked Against the Repository

**README.md — required contents, verified against actual current headings [VERIFIED FACT]:**

| Required | Present? | Section |
|---|---|---|
| Dataset description | Yes | `## Dataset`, `## Overview` |
| Problem statement | Yes | `## Problem statement` |
| Results summary table | Yes | `## Results` (filled with real numbers this session) |
| Environment setup | Yes | `## Installation` |
| How-to-run instructions | Yes | `## Usage` |

**GitHub repository structure — required vs. actual [VERIFIED FACT]:**

| Required | Present? |
|---|---|
| `README.md` | Yes |
| `requirements.txt` | Yes |
| `data/` (raw files or download script) | `data/raw/` and `data/processed/` exist; raw CSVs are intentionally `.gitignore`d (several GB) with download instructions in the README instead — an explicitly allowed alternative per the PDF ("Raw dataset file(s) **or** a script to download them"). |
| `notebooks/` | Present, but named `01_EDA.ipynb`/`02_Classification.ipynb`/`03_Regression.ipynb`/`04_Clustering.ipynb` rather than the PDF's example `regression.ipynb`/`classification.ipynb`/`clustering.ipynb` — see the naming note in Part 7. |
| `models/` (optional) | Present, with per-track subfolders; actual `.joblib`/`.json` artifacts are `.gitignore`d (regenerated by running the notebooks) |
| `app/` (if attempting bonus) | Present — `app/streamlit_app.py` |

**Commit history — "meaningful milestones, not a single bulk upload" [VERIFIED FACT, and a real issue to flag]:**

```
9e6d61d  2026-08-04  Initial commit: CyberSentinel SIEM ML pipeline
0fed4b5  2026-09-29  Update ML notebooks, preprocessing, and README   (3116 insertions, 483 deletions)
06e4605  2026-09-29  Update ML notebooks, preprocessing, and README   (711 insertions, 1072 deletions)
2e14452  2026-09-29  Update ML notebooks, preprocessing, and README   (697 insertions)
```

There **are** 4 distinct commits (not a single bulk upload), so the letter of "not one commit" is satisfied. However, **the 3 most recent commits share an identical, generic message** rather than descriptive milestone messages like the PDF's own examples ("Add Random Forest with GridSearchCV", "Complete clustering EDA"). This is worth improving — **[ACTION NEEDED]** either make future commits genuinely descriptive, or consider whether the team wants to present this history as-is and be ready to explain it in viva if asked. This is not something to leave unaddressed and assume is fine.

---

## 16. Bonus — GUI / Deployment

**[VERIFIED FACT / NOT VERIFIED]**
- `app/streamlit_app.py` **exists** (26,659 bytes) and, by inspection, reads saved model artifacts from `models/{classification,regression,clustering}/` and presents a SOC-style dashboard with a detection page, model analytics, and a threat-intelligence page.
- **Not verified this session:** whether the app actually launches and runs without errors — this documentation task did not include starting the Streamlit server. **[ACTION NEEDED]** the team should run `streamlit run app/streamlit_app.py` themselves and confirm it works end-to-end before claiming the +1 GUI bonus.
- **Public deployment:** no evidence of deployment configuration (no Streamlit Cloud/HF Spaces/Render config files, no deployed URL mentioned anywhere in the repo). **Status: not present.** The +1 deployment bonus and the +2 combined bonus are **not currently claimable** unless the team deploys it themselves.

---

## 17. Three-Member Team Structure (by complete track, per the official rule)

*(Names to be filled in — the PDF requires ownership by complete track, explicitly warning against splitting "one person does EDA, one does models, one does slides.")*

### Member 1 — [Name to fill]
**Primary track:** `01_EDA.ipynb` (feeds every other track) + the shared `src/` modules.
**Secondary responsibilities:** understanding how EDA findings (imbalance, correlation, port leakage) drove decisions in the other three notebooks.
**Must explain line-by-line:** the cleaning pipeline, the outlier-check justification, the `header_payload_ratio` feature end-to-end, and every EDA visualization.
**Likely viva questions:** "Why did you check outliers but not remove them?" / "Walk us through how `header_payload_ratio` is calculated and why." / "Why is `destination_port` kept as a feature despite the leakage risk you found?"

### Member 2 — [Name to fill]
**Primary track:** `02_Classification.ipynb` (all 10 algorithms, Part A and Part B).
**Secondary responsibilities:** the EDA imbalance finding, since it directly justifies this track's design.
**Must explain line-by-line:** all 10 classifiers at a basic level, the SVM shared-test-split fix, the full comparison table, SMOTE's effect, and the 3-model tuning results (including why MLP got worse).
**Likely viva questions:** "Why weighted F1 and not just accuracy?" / "What does AdaBoost's low macro-F1 tell you?" / "Explain One-vs-Rest ROC-AUC." / "Why did SMOTE barely change weighted F1 but change macro F1?"

### Member 3 — [Name to fill]
**Primary track:** `03_Regression.ipynb` + `04_Clustering.ipynb`.
**Secondary responsibilities:** how the risk score connects to EDA's behavioral features; how clustering's correlation-pruning connects to EDA's correlation analysis.
**Must explain line-by-line:** how the custom risk score is built (and why it's not a label lookup), all 10 regressors, the regression tuning results, K-Means vs. Agglomerative, the "average linkage rejected despite higher silhouette" finding, and cluster interpretation.
**Likely viva questions:** "Why isn't the risk score just a label lookup?" / "Why did you reject the linkage method with the better silhouette score?" / "Why can't you compare K-Means and Agglomerative's metrics directly?"

*(Rationale for pairing Regression + Clustering under one member: EDA is unusually large since it underpins every other notebook, and Classification alone is the most complex single track — 10 algorithms, SMOTE, and an explainability/SHAP section.)*

---

## 18. What All 3 Members Must Know

- **Project objective:** classify attack type, predict a risk score, and cluster behavior — from CICIDS2017 flow data.
- **Dataset:** CICIDS2017, 2,520,798 cleaned rows, 15 classes, severe imbalance.
- **Complete pipeline:** Part 6 above.
- **Preprocessing:** train-only-fit scaling, stratified 80:20 split, same split reused per track.
- **Feature engineering:** `header_payload_ratio` — what it is and why.
- **Basic idea of every algorithm:** Part 3 tables.
- **Evaluation metrics:** Accuracy/F1/confusion matrix/ROC-AUC (classification); R²/RMSE/MAE (regression); Silhouette/Davies-Bouldin/Calinski-Harabasz (clustering).
- **Important results:** Bagging won classification (F1 wtd 0.9873); Random Forest won regression (R² 0.9993 tuned); K-Means found k=10, several clusters behaviorally pure.
- **Limitations:** sampled (not full 2.52M-row) training; `destination_port` leakage risk; near-ceiling regression R² partly a validation artifact.
- **GitHub structure:** present, but recent commit messages are generic — see Part 15.
- **README:** all 5 required elements present.
- **Deployment/GUI status:** GUI code exists, **not yet verified to run**; no public deployment exists.

---

## 19. Viva Question Bank

**Basic project**
- *What is your project?* "An ML pipeline on CICIDS2017 network-traffic data covering all three required tracks: regression (risk score), classification (attack type), and clustering (behavior groups)."
- *Why this dataset?* "It's a real, published cybersecurity dataset with genuine captured attacks, at the flow level, which suits classical ML."

**Dataset**
- *What does one row represent?* "One network flow — a summarized connection, not a packet."
- *How imbalanced is it?* "190,459.7-to-1, BENIGN vs. Heartbleed."
- *What data-quality issues did you find?* "A negative flow duration value, a duplicate-looking column pair, and a text-encoding glitch in three label names — all documented, none silently fixed without saying so."

**Cybersecurity**
- *What is a DoS attack?* "Flooding a target so it can't serve real users." *DDoS?* "Same idea, from many sources at once."
- *What does destination_port tell you?* "The service being targeted — and in our data, a strong but risky signal, since most attack types sit on one fixed port."

**EDA**
- *Why a correlation heatmap?* "To catch redundant features before they distort distance-based clustering."
- *Why check outliers if you're not removing them?* "To make an informed, justified decision instead of blindly dropping or blindly keeping them — we found they're often the actual attack signal here."

**Preprocessing**
- *What is data leakage, concretely, in your project?* "If we'd fit the scaler or the risk-score's percentile boundaries on the full dataset instead of the training split only, information from the test set would leak into training — we specifically avoid this everywhere."
- *Why stratified split?* "So rare classes like Heartbleed (11 rows) actually appear in both splits."

**Feature engineering**
- *Walk us through `header_payload_ratio`.* "(header bytes) divided by (payload bytes) — high when a flow is mostly overhead, almost no real data, which we found matches DoS Slowhttptest's actual behavior in our sample."

**Regression**
- *Why is Polynomial Regression's R² so low?* "It's restricted to only the 5 risk-score indicator features, not all 79 — not a fair comparison to the other 9 models, and we say so explicitly."
- *What's the difference between R², RMSE, and MAE?* "R² is how much variance is explained (0–1); RMSE and MAE are both average error size, but RMSE punishes large errors more."
- *Why tune Random Forest and Decision Tree specifically?* "They were the top 2 by R² in the untuned comparison — the PDF requires tuning at least 2 models, and cross-validation for the top 2."

**Classification**
- *Why do you report macro-F1 alongside weighted F1?* "Weighted F1 can hide poor performance on rare classes behind BENIGN's volume; macro-F1 treats every class equally, which is how we caught AdaBoost's weakness."
- *Explain One-vs-Rest ROC-AUC.* "For each of the 15 classes, treat it as one-vs-everything-else, compute that ROC-AUC, then average across classes."
- *Why did MLP get worse after tuning?* "The search optimizes a cross-validated training-data score, which doesn't always align perfectly with the specific held-out test set's score — a real, honestly-reported outcome, not a bug."

**Clustering**
- *Why are labels not used until the end?* "Clustering is meant to discover structure blind to labels; we only bring labels back afterward, purely to describe what a cluster turned out to represent."
- *Why did you reject 'average' linkage despite its higher silhouette score?* "It put 99.1% of points in one cluster — a degenerate, useless split. Silhouette alone doesn't catch that; we added a balance check specifically because of this."

**Hyperparameter tuning**
- *What's the difference between GridSearchCV and RandomizedSearchCV, and which did you use?* "GridSearchCV tries every combination; RandomizedSearchCV samples a fixed number randomly — we used RandomizedSearchCV everywhere, since exhaustive grids would be too slow across this many models."
- *What is cross-validation, concretely?* "Splitting training data into folds and rotating which fold is held out, to get a score less dependent on one lucky/unlucky split — we used `cv=3` inside tuning, plus a separate 5-fold check on the top-2 models."

**Metrics**
- *What does a Davies-Bouldin index of 0.899 mean?* "Lower is better; it measures how similar each cluster is to its most-similar neighboring cluster."

**Data leakage**
- *Give a concrete example from your own project where leakage could have happened but didn't.* "If we'd fit `StandardScaler` before splitting into train/test, the scaler's mean/std would already reflect test rows — we split first, then fit the scaler on the training rows only."

**GitHub**
- *Does your repo have a meaningful commit history?* "Yes, 4 distinct commits, though the 3 most recent share a generic message rather than being individually descriptive — something we're aware of and can improve." (Answer honestly if asked — don't claim it's perfect.)

**Deployment**
- *Do you have a working GUI?* "The code for a Streamlit dashboard exists and reads our saved models, but we haven't yet verified it runs end-to-end / haven't deployed it publicly." (Only claim more once the team has actually tested it.)

**Limitations**
- *What's the biggest limitation of your regression track?* "The R² looks almost perfect partly because the risk score is a deterministic function of the same input features — it measures how well we can reconstruct our own formula, not real-world risk-assessment accuracy against independent ground truth."

**Future improvements**
- *What would you do with more time?* "Retrain the classifier without `destination_port` as a robustness check; scale training to the full 2.52M rows with a more scalable algorithm; verify and deploy the Streamlit app for the bonus marks."

---

## 20. Final 5-Minute Revision

- **Project:** CyberThreat-ML — intrusion detection, risk scoring, and behavioral clustering on network flow data.
- **Dataset:** CICIDS2017 (Canadian Institute for Cybersecurity).
- **Rows:** 2,830,743 raw → 2,520,798 cleaned.
- **Features:** 78 raw → 79 after engineering.
- **Target:** `label` (classification); custom risk score (regression).
- **Classes:** 15.
- **Regression target:** 0–100 behavior-based risk score (not a label lookup).
- **Engineered feature:** `header_payload_ratio`.
- **10 regression models:** Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN Regression.
- **5 Part-A classifiers:** Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVM.
- **5 Part-B classifiers:** Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP.
- **2 clustering models:** K-Means, Agglomerative Hierarchical Clustering.
- **Main preprocessing:** train-only-fit scaling, stratified 80:20 split, IQR outlier check (kept), correlation pruning (clustering only).
- **Metrics:** R²/RMSE/MAE; Accuracy/Precision/Recall/F1/ROC-AUC/confusion matrix; Silhouette/Davies-Bouldin/Calinski-Harabasz.
- **Tuning:** `RandomizedSearchCV` + cross-validation, on Random Forest/Decision Tree (regression) and Random Forest/SVM/MLP (classification).
- **Important visualizations:** 79-feature distribution grid, correlation heatmap, 2 scatter plots, residual/actual-vs-predicted plots, confusion-matrix grid, elbow curve, dendrogram, PCA/t-SNE cluster plots.
- **Main findings:** Bagging won classification (F1 wtd 0.9873); Random Forest won regression (R² 0.9993 tuned); k=10 clusters, several behaviorally pure; severe class imbalance; real `destination_port` leakage risk.
- **Main limitations:** stratified-sample training, not full dataset; regression R² partly a validation artifact; GUI not yet verified/deployed.
- **GUI/deployment:** Streamlit app code exists, not yet verified running, not publicly deployed.
- **Member 1:** EDA & Preprocessing + shared `src/`. **Member 2:** Classification. **Member 3:** Regression + Clustering.

---

## 21. Academic Integrity Note

Per the official PDF: AI may assist with **code scaffolding**, not analysis or interpretation, and any AI assistance must be cited in the README (already done — see `README.md`'s "AI assistance disclosure" section).

This document mixes three kinds of content, labelled throughout:
- **[VERIFIED FACT]** — objectively checked against the code/notebooks/execution output.
- **[ML CONCEPT]** — general algorithm/metric explanations, not project-specific claims.
- **[TEAM TO CONFIRM]** — an honest reading of the data, but genuinely the team's job to personally verify, understand, and be able to defend as their own interpretation in viva — not to repeat verbatim as if it were spontaneously their own idea without having actually looked at the underlying plot/table themselves.

Do not present the **[TEAM TO CONFIRM]** items in viva without having personally looked at the referenced plot or table first.

---

## 22. Final Audit — Every Official Requirement vs. Current Status

| Requirement | Official PDF requirement | Current status | Evidence / location | Action needed |
|---|---|---|---|---|
| A1 (Review 1) | Shape, dtypes, missing-value counts, class distribution | **COMPLETE** | `01_EDA.ipynb` cells 8, 9, 11, 19 | None |
| A2 (Review 1) | Distribution plot per feature, correlation heatmap, target distribution, ≥2 scatter plots | **COMPLETE** | `01_EDA.ipynb` cells 21, 28, 30–31, 33 | None |
| A3 (Review 1) | Written observation per major plot | **COMPLETE** | Markdown cells after every plot, `01_EDA.ipynb` | None |
| B1 (Review 1) | Missing values, duplicates, outliers checked & treated with justification | **COMPLETE** | `01_EDA.ipynb` cells 11–13; `src/preprocessing.py: clean_data(), check_outliers()` | None |
| B2 (Review 1) | Categorical encoding, correct scaler (train-only fit), stratified split | **COMPLETE** | `src/preprocessing.py: scale_features(), split_data()`; no categorical features exist besides the label | None |
| B3 (Review 1) | ≥1 engineered feature with written justification | **COMPLETE** | `header_payload_ratio`, `src/preprocessing.py: engineer_features()`, justified in `01_EDA.ipynb` | None |
| C1 (Review 1) | All 10 regression algorithms trained, predict, no errors | **COMPLETE** | `03_Regression.ipynb`, verified 0 errors this session | None |
| C2 (Review 1) | Single table, R²/RMSE/MAE, all 10, same test split, ranked by R² | **COMPLETE** | `03_Regression.ipynb` comparison table; SVR fixed to shared test split | None |
| C3 (Review 1) | Tuning on ≥2 models, best params + improvement reported | **COMPLETE** | `03_Regression.ipynb` — added from scratch this session (was previously missing entirely) | None |
| C4 (Review 1) | Residual + actual-vs-predicted plots; tree feature importance | **COMPLETE** | `03_Regression.ipynb` | None |
| D1/Part A (Review 1) | All 5 Part-A algorithms trained, predict, no errors | **COMPLETE** | `02_Classification.ipynb`; SVM fixed to shared test split | None |
| D2/Part A (Review 1) | Accuracy, weighted F1, confusion matrix per algorithm; preliminary table | **COMPLETE** | `02_Classification.ipynb` | None |
| E1 (Review 1) | Clear narrative; all members can explain | **COMPLETE (doc), TEAM TO VERIFY (viva-readiness)** | Markdown headings + final summary sections added this session in all 4 notebooks | Team must actually rehearse — a document can't verify understanding |
| Part B A1 (Review 2) | All 5 Part-B algorithms trained on same dataset as Part A, no errors | **COMPLETE** | `02_Classification.ipynb`, same `working_df`/split as Part A | None |
| Part B A2 (Review 2) | Single table, all 10, Accuracy/Precision/Recall/F1/ROC-AUC | **COMPLETE** | `02_Classification.ipynb` comparison table includes all 5 required metrics (plus macro-F1 as a bonus column) | None |
| Part B A3 (Review 2) | Best model justified; tuning applied; improvement in ≥1 metric documented | **COMPLETE** | Final-selection section, `02_Classification.ipynb` — Bagging justified as overall winner across 14 candidates | None |
| B1 Clustering (Review 2) | Both algorithms fitted, cluster labels obtained, no errors | **COMPLETE** | `04_Clustering.ipynb` | None |
| B2 Clustering (Review 2) | Silhouette/DB/CH for all algorithms; elbow curve; dendrogram | **COMPLETE** | `04_Clustering.ipynb` | None |
| B3 Clustering (Review 2) | PCA 2D plot for every algorithm; t-SNE for ≥1 | **COMPLETE** | `04_Clustering.ipynb`: `kmeans_pca_2d.png`, `agglomerative_pca_2d.png`, `agglomerative_tsne.png` | None |
| C1 Pipeline (Review 2) | Notebooks run top-to-bottom, no errors; modular; commented | **COMPLETE** | Verified via clean-kernel execution this session; `src/` modules used throughout | None |
| C2 README (Review 2) | Dataset description, problem statement, results table, setup, how-to-run | **COMPLETE** | See Part 15 table above | None |
| C3 GitHub (Review 2) | Meaningful commit history; requirements.txt present | **PARTIAL** | 4 real commits exist (not a bulk upload), but 3 most recent share an identical generic message | Make future commits descriptive; consider whether to address the existing ones |
| D1 Presentation (Review 2) | Story arc, clear visualisations, well-organised | **COMPLETE (doc)** | This guide + notebook Markdown narrative | Team must build/rehearse actual slides or notebook walkthrough |
| D2 Viva (Review 2) | All members demonstrate understanding | **NOT VERIFIABLE FROM CODE** | — | Team must rehearse; no document can certify this |
| 7.1 General | `random_state=42` everywhere applicable | **COMPLETE** | Verified via grep across `src/` and all notebooks | None |
| 7.2 General | Consistent split per track; CV for top-2; summary tables not scattered prints | **COMPLETE** | Fixed this session (SVM/SVR were the one violation found) | None |
| 7.3 General | Titles/labels/legends; `tight_layout`; colourblind palettes | **COMPLETE, one noted deviation** | All plots comply; new 15-class scatter plots use `tab20` (not on the literal `tab10`/`Set2`/`colorblind` list, but a reasonable choice for 15 categories) | None required; could switch to `tab10` with repeated colors if the instructor is strict about the literal palette list |
| 7.4 Team | One primary owner per complete track | **COMPLETE (assigned in this doc)** | Part 17 | Team must fill in real names and actually follow the assigned ownership |
| 7.5 Academic integrity | External code cited; original analysis; AI usage cited in README | **COMPLETE, with a caveat** | `README.md` "AI assistance disclosure" section added this session; no external-sourced code found anywhere in the project | Team must personally review all **[TEAM TO CONFIRM]**-tagged interpretations in this document and in the notebooks before presenting them as their own |
| Bonus +1 GUI | Working web interface returning predictions | **NOT VERIFIED** | `app/streamlit_app.py` exists, reads saved models, but was not run this session | Run `streamlit run app/streamlit_app.py` and confirm it works |
| Bonus +1 Deployment | Publicly accessible URL | **NOT PRESENT** | No deployment config or URL found anywhere in the repo | Deploy if the team wants the bonus marks |

### What is completely ready
Every graded rubric line item across both reviews — Sections A/B/C/D of Review 1 and Sections A/B/C of Review 2 — is implemented, executes with zero errors (verified via a clean-kernel run this session), and is documented with real, non-fabricated results.

### What is still missing
- Descriptive, per-milestone commit messages (currently 3 recent commits share one generic message).
- Verified, working Streamlit GUI (code exists, untested this session).
- Public deployment (does not exist).

### What must be personally checked before submission
- Every **[TEAM TO CONFIRM]**-tagged claim in this document (feature-engineering evidence, cluster interpretation, EDA observations) — look at the actual plot/table yourselves.
- That the Streamlit app actually runs (`streamlit run app/streamlit_app.py`).
- Real team member names filled into Part 17/Part 20 in place of the placeholders.

### What each member must study first
- **Member 1:** `01_EDA.ipynb` top to bottom, plus `src/preprocessing.py`.
- **Member 2:** `02_Classification.ipynb` top to bottom, especially the tuning and SMOTE sections.
- **Member 3:** `03_Regression.ipynb` and `04_Clustering.ipynb` top to bottom, especially `src/risk_score.py` and the linkage-comparison logic.
