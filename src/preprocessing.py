"""
preprocessing.py
================
Shared, reusable preprocessing pipeline for CyberThreat-ML.

Every notebook (EDA, classification, regression, clustering) imports from
here instead of repeating "load -> clean -> encode -> scale -> split"
logic. This keeps preprocessing identical and reproducible across all
three tracks, and is the single place changes need to be made.

Design rules followed throughout:
- random_state=42 everywhere a seed is needed
- scalers/encoders are FIT on training data only, then applied to test
  data, to avoid data leakage
- nothing here silently drops rows without reporting how many and why
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_STATE = 42

# Pure identifiers (not behavioral features) -- dropped whenever present.
# NOTE: "destination_port" is deliberately NOT in this list. It's a known
# potential shortcut feature (models can learn "port 21/22 = brute force"
# without learning real traffic behavior) -- see the classification
# notebook for the port-leakage ablation study discussed during planning,
# rather than silently dropping it here.
IDENTIFIER_COLUMNS = [
    "flow_id", "source_ip", "src_ip", "destination_ip", "dst_ip",
    "timestamp", "source_port", "src_port",
]


def load_and_merge_csvs(raw_dir):
    """
    Load every CSV file found under raw_dir (recursively) and concatenate
    into one DataFrame.

    Returns
    -------
    merged : pd.DataFrame
    report : dict of {filename: row_count}, plus '_TOTAL_' and
             '_FILES_MERGED_' -- printed in the EDA notebook so the merge
             is auditable rather than a black box.
    """
    csv_paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found under {raw_dir}. "
            "Extract MachineLearningCSV.zip into data/raw/ first."
        )

    frames = []
    report = {}
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        report[os.path.basename(path)] = len(df)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    report["_FILES_MERGED_"] = len(csv_paths)
    report["_TOTAL_"] = len(merged)
    return merged, report


def standardize_columns(df):
    """
    CICIDS2017's CSVs are notorious for inconsistent column names across
    day-files (leading/trailing spaces, mixed case, stray punctuation).
    This strips all of that down to a consistent lower_snake_case so every
    downstream function can rely on stable names regardless of which
    day-file a column came from.

    Example: ' Flow Bytes/s' -> 'flow_bytess', ' Destination Port ' -> 'destination_port'
    """
    df = df.copy()
    new_cols = []
    for col in df.columns:
        c = str(col).strip()
        c = re.sub(r"[^\w\s]", "", c)       # drop punctuation like '/'
        c = re.sub(r"\s+", "_", c).lower()  # spaces -> underscores, lowercase
        new_cols.append(c)
    df.columns = new_cols
    return df


def clean_data(df, verbose=True):
    """
    Applies, in order:
      1. Replace +/-Infinity with NaN (common in flow_bytes_s / flow_packets_s
         when flow_duration is 0)
      2. Drop rows with any remaining NaN (reported, not silent)
      3. Drop exact duplicate rows
      4. Drop pure-identifier columns if present (see IDENTIFIER_COLUMNS)

    Returns the cleaned DataFrame and a report dict of what was removed,
    so the EDA notebook can *show* these numbers rather than assert them.
    Call standardize_columns() before this.
    """
    df = df.copy()
    report = {"rows_before": len(df)}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols].to_numpy()).any(axis=1)
    report["rows_with_infinity"] = int(inf_mask.sum())
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    nan_mask = df.isna().any(axis=1)
    report["rows_with_nan"] = int(nan_mask.sum())
    df = df.dropna()

    before_dupes = len(df)
    df = df.drop_duplicates()
    report["duplicate_rows_removed"] = before_dupes - len(df)

    dropped_cols = [c for c in df.columns if c in IDENTIFIER_COLUMNS]
    if dropped_cols:
        df = df.drop(columns=dropped_cols)
    report["identifier_columns_dropped"] = dropped_cols
    report["rows_after"] = len(df)

    if verbose:
        print(f"Rows before cleaning:        {report['rows_before']:,}")
        print(f"Rows with Infinity:          {report['rows_with_infinity']:,}")
        print(f"Rows with NaN:               {report['rows_with_nan']:,}")
        print(f"Duplicate rows removed:      {report['duplicate_rows_removed']:,}")
        print(f"Identifier columns dropped:  {dropped_cols}")
        print(f"Rows after cleaning:         {report['rows_after']:,}")

    return df, report


def encode_labels(y):
    """
    Label-encode the attack category target. Returns the encoded array
    AND the fitted LabelEncoder, so predictions can be mapped back to
    human-readable class names later (e.g. in the Streamlit app).
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def split_data(X, y, test_size=0.2, stratify=True, random_state=RANDOM_STATE):
    """Stratified train/test split -- stratification keeps rare attack
    classes proportionally represented in both splits."""
    stratify_arg = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )


def scale_features(X_train, X_test):
    """
    Fit StandardScaler on TRAINING data only, then transform both splits.
    Fitting on the full dataset before splitting is a classic leakage bug;
    requiring the split first structurally prevents that here.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def check_outliers(df, columns=None, iqr_multiplier=1.5):
    """
    Report (does not remove) statistical outliers per numeric column using
    the IQR rule: values outside [Q1 - k*IQR, Q3 + k*IQR].

    Outliers are reported, not dropped, by design: in network-flow
    intrusion data, extreme values (packet-flood rates, near-zero flow
    durations) are frequently the attack signal this project exists to
    detect, not measurement error. Blanket removal would silently delete
    rare-attack rows on top of the imbalance the dataset already has.
    Genuinely invalid values (Infinity/NaN) are already handled in
    clean_data(); this function is for the remaining, valid-but-extreme
    values, so a track's notebook can make an informed, justified call
    about whether to treat them (e.g. via scaling / tree-based models
    that are naturally robust to them) rather than dropping rows here.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    rows = []
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        n_out = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append({
            "feature": col,
            "n_outliers": n_out,
            "pct_outliers": round(100 * n_out / len(df), 2),
            "lower_bound": lower,
            "upper_bound": upper,
        })
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False).reset_index(drop=True)


def engineer_features(df):
    """
    Add engineered feature(s) on top of the raw CICFlowMeter columns.

    `header_payload_ratio` = (fwd_header_length + bwd_header_length) /
    (total_length_of_fwd_packets + total_length_of_bwd_packets). Legitimate
    bulk-data flows carry large payloads relative to their fixed per-packet
    header overhead, so this ratio is small. Reconnaissance-style traffic
    (port scans, SYN floods) is dominated by packets with little or no
    payload, so headers make up most of a flow's bytes and this ratio
    spikes -- a lightweight, interpretable probing signal.

    None of the 78 raw columns expresses this relationship directly
    (header and payload lengths only exist as separate columns), and it is
    orthogonal to the packet-rate / byte-rate / flag-anomaly / duration /
    traffic-asymmetry indicators src/risk_score.py already derives for the
    regression target, so it contributes new signal rather than restating
    it. Added here (not per-notebook) so it's computed once and reaches
    every track through the shared cleaned_dataset.csv.
    """
    df = df.copy()
    header_cols = [c for c in ("fwd_header_length", "bwd_header_length") if c in df.columns]
    payload_cols = [c for c in ("total_length_of_fwd_packets", "total_length_of_bwd_packets") if c in df.columns]
    if len(header_cols) == 2 and len(payload_cols) == 2:
        header_total = df[header_cols[0]] + df[header_cols[1]]
        payload_total = df[payload_cols[0]] + df[payload_cols[1]]
        # +1 avoids div-by-zero for zero-payload flows without needing a
        # separate branch; header/payload byte counts are non-negative.
        df["header_payload_ratio"] = (header_total / (payload_total + 1)).round(4)
    return df


def stratified_sample(df, label_col, max_total=200_000, min_per_class=200, random_state=RANDOM_STATE):
    """
    Build a stratified subsample that keeps EVERY class fully if it's
    smaller than min_per_class, while capping dominant classes (BENIGN
    especially) so the heaviest algorithms (SVM, MLP, hyperparameter
    search) stay tractable on a laptop. Rare attack classes are never
    thinned out by this function.

    This reduces the full ~2.8M-row dataset to a manageable working set
    BEFORE the train/test split -- both splits then reflect this
    (still-imbalanced-but-tractable) distribution.
    """
    counts = df[label_col].value_counts()
    n_classes = len(counts)
    target_per_class = max(min_per_class, max_total // n_classes)

    parts = []
    for cls, n in counts.items():
        take = min(n, target_per_class)
        parts.append(df[df[label_col] == cls].sample(n=take, random_state=random_state))

    sampled = pd.concat(parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
