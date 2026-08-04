"""
risk_score.py
=============
Behavior-based cybersecurity risk score (0-100), built from network flow
characteristics rather than a hardcoded label-to-number lookup
(e.g. BENIGN=0, DDoS=100), per the project requirement.

Approach
--------
1. Derive behavioral indicators the IDS literature associates with
   malicious traffic: packet rate, byte rate, connection-flag anomalies
   (SYN without matching ACK/FIN -- half-open connections; RST-heavy --
   failed connections), flow-duration extremity, and forward/backward
   traffic asymmetry.
2. Normalize each indicator to [0, 1] using percentile ranks FIT ON
   TRAINING DATA ONLY (same no-leakage discipline as scaling/encoding
   elsewhere) -- percentile ranks are far more robust to the
   Infinity/outlier issues in this dataset than plain min-max scaling.
3. Combine with domain-informed weights into a single 0-100 score.
4. Bin into Low (0-20) / Medium (20-60) / High (60-100).

The default weights are a documented starting point (packet-rate flooding
and flag anomalies weighted highest, as the strongest malicious-behavior
signals in the literature; duration and traffic-shape asymmetry as
secondary signals). They get sanity-checked against real percentile
distributions -- and against known labels for VALIDATION ONLY, never as
an input to the formula -- in 03_regression.ipynb once real data is loaded.
"""

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "packet_rate": 0.25,
    "byte_rate": 0.20,
    "flag_anomaly": 0.25,
    "duration_extremity": 0.15,
    "traffic_asymmetry": 0.15,
}


class RiskScoreCalculator:
    """
    Fit percentile-normalization boundaries on training data, then score
    any batch of flows (train, test, or a brand-new upload in the
    Streamlit app) on a consistent 0-100 scale.
    """

    def __init__(self, weights=None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._fitted = False
        self._percentile_refs = {}

    @staticmethod
    def _safe_col(df, *candidates):
        """Return the first candidate column name present in df. Handles
        minor naming drift across dataset releases / cleaning steps."""
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _compute_raw_indicators(self, df):
        """Derive the five raw behavioral indicators from flow columns.
        Returns a DataFrame of raw (not yet normalized) indicator values.
        Missing source columns fall back to 0.0 for that indicator rather
        than raising, so scoring degrades gracefully on partial uploads."""
        out = pd.DataFrame(index=df.index)

        col_pkt_rate = self._safe_col(df, "flow_packetss", "flow_packets_s")
        col_byte_rate = self._safe_col(df, "flow_bytess", "flow_bytes_s")
        col_duration = self._safe_col(df, "flow_duration")
        col_syn = self._safe_col(df, "syn_flag_count")
        col_ack = self._safe_col(df, "ack_flag_count")
        col_fin = self._safe_col(df, "fin_flag_count")
        col_rst = self._safe_col(df, "rst_flag_count")
        col_fwd_pkts = self._safe_col(df, "total_fwd_packets")
        col_bwd_pkts = self._safe_col(df, "total_backward_packets", "total_bwd_packets")

        out["packet_rate"] = df[col_pkt_rate] if col_pkt_rate else 0.0
        out["byte_rate"] = df[col_byte_rate] if col_byte_rate else 0.0

        if col_syn and col_ack and col_fin:
            established = df[col_ack] + df[col_fin] + 1e-6
            out["flag_anomaly"] = df[col_syn] / established
        elif col_rst:
            out["flag_anomaly"] = df[col_rst]
        else:
            out["flag_anomaly"] = 0.0

        if col_duration:
            # Distance from the median flow duration in log space (flow
            # durations are heavily right-skewed, so raw distance would
            # be dominated by a handful of very long flows).
            log_dur = np.log1p(df[col_duration].clip(lower=0))
            out["duration_extremity"] = (log_dur - log_dur.median()).abs()
        else:
            out["duration_extremity"] = 0.0

        if col_fwd_pkts and col_bwd_pkts:
            total = df[col_fwd_pkts] + df[col_bwd_pkts] + 1e-6
            out["traffic_asymmetry"] = (df[col_fwd_pkts] - df[col_bwd_pkts]).abs() / total
        else:
            out["traffic_asymmetry"] = 0.0

        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def get_indicators(self, df):
        """
        Public accessor for the five raw (not yet normalized) behavioral
        indicators -- useful when a notebook wants the indicators
        themselves as a compact feature set (e.g. Polynomial Regression,
        where expanding all ~78 raw features to degree 2 would be far too
        many terms, but expanding these 5 is cheap and conceptually
        meaningful: their pairwise interactions are exactly the kind of
        "packet rate x flag anomaly" combinations a security analyst
        would reason about by hand).
        """
        return self._compute_raw_indicators(df)

    def fit(self, df):
        """Store percentile reference distributions from TRAINING data."""
        raw = self._compute_raw_indicators(df)
        self._percentile_refs = {col: np.sort(raw[col].to_numpy()) for col in raw.columns}
        self._fitted = True
        return self

    @staticmethod
    def _percentile_normalize(values, sorted_ref):
        """Rank `values` against the fitted (sorted) reference distribution,
        returning a 0-1 percentile score -- robust to outliers/Infinity,
        unlike plain min-max scaling."""
        ranks = np.searchsorted(sorted_ref, values, side="right")
        return np.clip(ranks / len(sorted_ref), 0.0, 1.0)

    def score(self, df):
        """Return a DataFrame with the raw indicators, the composite
        0-100 risk_score, and the Low/Medium/High risk_category."""
        if not self._fitted:
            raise RuntimeError("Call .fit(training_df) before .score(...)")

        raw = self._compute_raw_indicators(df)
        normalized = pd.DataFrame(index=df.index)
        for col in raw.columns:
            normalized[col] = self._percentile_normalize(
                raw[col].to_numpy(), self._percentile_refs[col]
            )

        composite = sum(normalized[col] * w for col, w in self.weights.items())
        result = raw.copy()
        result["risk_score"] = (composite * 100).round(2)
        result["risk_category"] = pd.cut(
            result["risk_score"], bins=[-0.1, 20, 60, 100], labels=["Low", "Medium", "High"]
        )
        return result
