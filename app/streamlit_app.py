"""
app/streamlit_app.py
=====================
CyberThreat-ML -- SOC-style dashboard for the intrusion detection system.

Run with:
    streamlit run app/streamlit_app.py

Pages:
  - Dashboard          : KPI cards, attack distribution, timeline, risk levels, recent events
  - Detection           : upload network traffic CSV -> attack type, confidence, risk score
  - Analytics           : model comparison tables from the classification/regression notebooks
  - Explainability      : feature importance / SHAP for the trained classifier
  - Threat Intelligence : what the clustering notebook discovered, mapped back to attack types

This file only READS artifacts produced by the notebooks (models/*, data/processed/*).
It never trains anything itself, and every page degrades gracefully with a clear
message if the notebook that produces its data hasn't been run yet.
"""

import os
import sys
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
import preprocessing as prep  # noqa: E402
import risk_score as rs  # noqa: E402

CLS_DIR = os.path.join(BASE_DIR, "models", "classification")
REG_DIR = os.path.join(BASE_DIR, "models", "regression")
CLU_DIR = os.path.join(BASE_DIR, "models", "clustering")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

st.set_page_config(
    page_title="CyberThreat-ML | SOC Dashboard",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME: dark SOC/security-operations palette injected via CSS.
# Streamlit doesn't allow full theme control from Python, so we override
# the relevant elements directly. Severity colors are used consistently
# everywhere a risk category or alert level appears.
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0A0E17",
    "surface": "#111827",
    "surface_2": "#1A2332",
    "border": "#232F42",
    "text": "#E5E9F0",
    "text_muted": "#8B95A8",
    "accent": "#22D3EE",
    "critical": "#EF4444",
    "high": "#F97316",
    "medium": "#F59E0B",
    "low": "#22C55E",
    "info": "#3B82F6",
}

RISK_COLORS = {"Low": COLORS["low"], "Medium": COLORS["medium"], "High": COLORS["critical"]}


def hex_to_rgba(hex_color, alpha=0.2):
    """Plotly properties like `fillcolor` need an rgba() string -- unlike
    CSS, they don't accept 8-digit hex (hex + alpha) values."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {{
        background-color: {COLORS['bg']};
        color: {COLORS['text']};
        font-family: 'Inter', sans-serif;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}
    h1, h2, h3, h4 {{ color: {COLORS['text']} !important; font-family: 'Inter', sans-serif; font-weight: 600; }}
    p, span, label, div {{ color: {COLORS['text']}; }}
    .stDataFrame, .stTable {{ font-family: 'JetBrains Mono', monospace; }}

    .soc-header {{
        display: flex; align-items: center; gap: 10px;
        padding-bottom: 4px;
    }}
    .live-dot {{
        height: 10px; width: 10px; border-radius: 50%;
        background-color: {COLORS['low']};
        box-shadow: 0 0 0 0 rgba(34,197,94,0.6);
        animation: pulse 2s infinite;
        display: inline-block;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }}
        70% {{ box-shadow: 0 0 0 8px rgba(34,197,94,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }}
    }}
    .kpi-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 18px 20px;
    }}
    .kpi-label {{
        color: {COLORS['text_muted']}; font-size: 13px; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .kpi-value {{
        font-family: 'JetBrains Mono', monospace; font-size: 30px; font-weight: 700;
        color: {COLORS['text']}; margin-top: 4px;
    }}
    .kpi-sub {{ font-size: 12px; color: {COLORS['text_muted']}; margin-top: 2px; }}
    .badge {{
        display: inline-block; padding: 3px 10px; border-radius: 4px;
        font-size: 12px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    }}
    .section-card {{
        background-color: {COLORS['surface']}; border: 1px solid {COLORS['border']};
        border-radius: 8px; padding: 16px 20px; margin-bottom: 16px;
    }}
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", accent=None):
    color = accent or COLORS["accent"]
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def severity_badge(category):
    color = RISK_COLORS.get(category, COLORS["info"])
    return (f'<span class="badge" style="background-color:{color}22; '
            f'color:{color}; border:1px solid {color}55;">{category}</span>')


# ---------------------------------------------------------------------------
# CACHED LOADERS -- each returns None (not an exception) if the notebook
# that produces its artifacts hasn't been run yet, so pages can show a
# clear instruction instead of crashing.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_classification_artifacts():
    required = ["best_model.joblib", "scaler.joblib", "label_encoder.joblib", "feature_names.json"]
    if not all(os.path.exists(os.path.join(CLS_DIR, f)) for f in required):
        return None
    model = joblib.load(os.path.join(CLS_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(CLS_DIR, "scaler.joblib"))
    encoder = joblib.load(os.path.join(CLS_DIR, "label_encoder.joblib"))
    with open(os.path.join(CLS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    info_path = os.path.join(CLS_DIR, "model_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    return {"model": model, "scaler": scaler, "encoder": encoder, "feature_names": feature_names, "info": info}


@st.cache_resource
def load_regression_artifacts():
    required = ["best_model.joblib", "scaler.joblib", "risk_score_calculator.joblib", "feature_names.json"]
    if not all(os.path.exists(os.path.join(REG_DIR, f)) for f in required):
        return None
    model = joblib.load(os.path.join(REG_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(REG_DIR, "scaler.joblib"))
    risk_calc = joblib.load(os.path.join(REG_DIR, "risk_score_calculator.joblib"))
    with open(os.path.join(REG_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    info_path = os.path.join(REG_DIR, "model_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    return {"model": model, "scaler": scaler, "risk_calc": risk_calc, "feature_names": feature_names, "info": info}


@st.cache_resource
def load_clustering_artifacts():
    required = ["kmeans_model.joblib", "scaler.joblib", "pca_transformer.joblib", "cluster_profile.csv"]
    if not all(os.path.exists(os.path.join(CLU_DIR, f)) for f in required):
        return None
    kmeans = joblib.load(os.path.join(CLU_DIR, "kmeans_model.joblib"))
    scaler = joblib.load(os.path.join(CLU_DIR, "scaler.joblib"))
    pca = joblib.load(os.path.join(CLU_DIR, "pca_transformer.joblib"))
    profile = pd.read_csv(os.path.join(CLU_DIR, "cluster_profile.csv"), index_col=0)
    dropped_path = os.path.join(CLU_DIR, "dropped_correlated_features.json")
    dropped = json.load(open(dropped_path)) if os.path.exists(dropped_path) else []
    return {"kmeans": kmeans, "scaler": scaler, "pca": pca, "profile": profile, "dropped_features": dropped}


@st.cache_data
def load_reference_sample(max_rows=5000):
    """A sample of the cleaned dataset, used only to populate the dashboard
    overview page with representative charts. Returns None if 01_EDA.ipynb
    hasn't been run yet."""
    path = os.path.join(PROCESSED_DIR, "cleaned_dataset.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False, nrows=200_000)
    label_col = "label" if "label" in df.columns else df.columns[-1]
    if len(df) > max_rows:
        df = prep.stratified_sample(df, label_col=label_col, max_total=max_rows, min_per_class=5)
    return df


@st.cache_data
def load_comparison_table(track):
    path = os.path.join(BASE_DIR, "models", track, "comparison_table.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# PREDICTION PIPELINE -- shared by the Dashboard's "recent events" table and
# the Detection page's file upload.
# ---------------------------------------------------------------------------
def predict_traffic(df_raw, cls_art, reg_art):
    """
    Standardize an uploaded (or reference) DataFrame, align it to the
    trained feature set, and return per-row predictions: attack type,
    confidence, risk score, risk category. Missing columns are filled with
    0 and reported, rather than silently guessed at or crashing.
    """
    df = prep.standardize_columns(df_raw.copy())
    label_col = "label" if "label" in df.columns else None
    df_features = df.drop(columns=[label_col]) if label_col else df

    warnings = []
    results = pd.DataFrame(index=df.index)

    if cls_art is not None:
        missing = [c for c in cls_art["feature_names"] if c not in df_features.columns]
        if missing:
            warnings.append(f"{len(missing)} expected column(s) missing from upload, filled with 0: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        X_aligned = df_features.reindex(columns=cls_art["feature_names"], fill_value=0)
        X_scaled = cls_art["scaler"].transform(X_aligned)
        pred_encoded = cls_art["model"].predict(X_scaled)
        results["predicted_attack"] = cls_art["encoder"].inverse_transform(pred_encoded)
        if hasattr(cls_art["model"], "predict_proba"):
            proba = cls_art["model"].predict_proba(X_scaled)
            results["confidence"] = (proba.max(axis=1) * 100).round(1)
        else:
            results["confidence"] = np.nan

    if reg_art is not None:
        scored = reg_art["risk_calc"].score(df)
        results["risk_score"] = scored["risk_score"].to_numpy()
        results["risk_category"] = scored["risk_category"].to_numpy()

    return results, warnings


def top_contributing_features(cls_art, n=5):
    model, names = cls_art["model"], cls_art["feature_names"]
    if hasattr(model, "feature_importances_"):
        s = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
        return s.head(n)
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        s = pd.Series(coef, index=names).sort_values(ascending=False)
        return s.head(n)
    return None


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
inject_theme()

with st.sidebar:
    st.markdown(f"""
    <div class="soc-header">
        <span style="font-size:22px;">:shield:</span>
        <span style="font-size:18px; font-weight:600;">CyberThreat-ML</span>
    </div>
    <div style="color:{COLORS['text_muted']}; font-size:12px; margin-bottom:16px;">
        Security Operations Console
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Dashboard", "Detection", "Analytics", "Explainability", "Threat Intelligence"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:12px; color:{COLORS['text_muted']};">
        <span class="live-dot"></span> &nbsp; Console active<br/>
        Last refresh: {datetime.now().strftime('%H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

cls_art = load_classification_artifacts()
reg_art = load_regression_artifacts()
clu_art = load_clustering_artifacts()


def not_trained_message(notebook_name):
    st.warning(
        f":warning: No trained model found yet. Run **notebooks/{notebook_name}** first -- "
        f"it saves the artifacts this page reads from `models/`."
    )


# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================
if page == "Dashboard":
    st.markdown("## Security Operations Dashboard")

    ref_df = load_reference_sample()
    if ref_df is None:
        st.warning(":warning: No processed data found yet. Run **notebooks/01_EDA.ipynb** first.")
    else:
        label_col = "label" if "label" in ref_df.columns else ref_df.columns[-1]

        # Score the reference sample so the dashboard has something real to show
        results, _ = predict_traffic(ref_df, cls_art, reg_art)

        total_events = len(ref_df)
        threat_count = int((ref_df[label_col] != "BENIGN").sum()) if "BENIGN" in ref_df[label_col].unique() else int(len(ref_df) * 0.3)
        critical_count = int((results["risk_category"] == "High").sum()) if "risk_category" in results.columns else 0
        avg_risk = results["risk_score"].mean() if "risk_score" in results.columns else np.nan

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Total events", f"{total_events:,}", "in current sample")
        with c2:
            kpi_card("Threats detected", f"{threat_count:,}", f"{threat_count / total_events * 100:.1f}% of traffic", accent=COLORS["medium"])
        with c3:
            kpi_card("Critical alerts", f"{critical_count:,}", "High risk category", accent=COLORS["critical"])
        with c4:
            kpi_card("Avg. risk score", f"{avg_risk:.1f}" if not np.isnan(avg_risk) else "N/A", "0-100 scale", accent=COLORS["accent"])

        st.markdown("<br/>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("#### Attack distribution")
            counts = ref_df[label_col].value_counts().reset_index()
            counts.columns = ["attack_type", "count"]
            fig = px.bar(counts, x="count", y="attack_type", orientation="h",
                         color="count", color_continuous_scale=["#1A2332", COLORS["accent"]])
            fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"],
                               font_color=COLORS["text"], showlegend=False, coloraxis_showscale=False,
                               margin=dict(l=0, r=0, t=10, b=0), height=320)
            st.plotly_chart(fig, width='stretch')

        with col_b:
            st.markdown("#### Risk level distribution")
            if "risk_category" in results.columns:
                risk_counts = results["risk_category"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
                fig = go.Figure(data=[go.Pie(
                    labels=risk_counts.index, values=risk_counts.values, hole=0.55,
                    marker=dict(colors=[RISK_COLORS[c] for c in risk_counts.index]),
                )])
                fig.update_layout(paper_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                                   margin=dict(l=0, r=0, t=10, b=0), height=320)
                st.plotly_chart(fig, width='stretch')
            else:
                not_trained_message("03_Regression.ipynb")

        st.markdown("#### Threat timeline")
        st.caption("Sequence-based view over the current sample (no live capture timestamps in the feature set by design -- see EDA notebook's identifier-column note).")
        bucketed = ref_df.reset_index(drop=True).copy()
        bucketed["sequence_bucket"] = bucketed.index // max(1, len(bucketed) // 40)
        bucketed["is_threat"] = (bucketed[label_col] != "BENIGN").astype(int) if "BENIGN" in bucketed[label_col].unique() else 0
        timeline = bucketed.groupby("sequence_bucket")["is_threat"].sum().reset_index()
        fig = px.area(timeline, x="sequence_bucket", y="is_threat")
        fig.update_traces(line_color=COLORS["accent"], fillcolor=hex_to_rgba(COLORS["accent"], 0.2))
        fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                           margin=dict(l=0, r=0, t=10, b=0), height=220,
                           xaxis_title="Event sequence", yaxis_title="Threats per bucket")
        st.plotly_chart(fig, width='stretch')

        st.markdown("#### Recent network events")
        display_cols = [label_col]
        if "predicted_attack" in results.columns:
            display_cols += ["predicted_attack", "confidence"]
        if "risk_score" in results.columns:
            display_cols += ["risk_score", "risk_category"]
        event_table = pd.concat([ref_df[[label_col]], results.drop(columns=[c for c in results.columns if c == label_col], errors="ignore")], axis=1)
        st.dataframe(event_table.head(25), width='stretch', height=320)

# ===========================================================================
# PAGE: DETECTION
# ===========================================================================
elif page == "Detection":
    st.markdown("## Threat Detection")
    st.caption("Upload a network traffic CSV (same column format as CICIDS2017) to classify each flow.")

    if cls_art is None and reg_art is None:
        not_trained_message("02_Classification.ipynb and 03_Regression.ipynb")
    else:
        uploaded = st.file_uploader("Network traffic CSV", type=["csv"])
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded, low_memory=False)
            st.write(f"Loaded {raw_df.shape[0]:,} rows x {raw_df.shape[1]} columns")

            results, warnings = predict_traffic(raw_df, cls_art, reg_art)
            for w in warnings:
                st.info(w)

            if len(results) == 1 or st.checkbox("Show single-row detail view", value=(len(results) <= 1)):
                row = results.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1:
                    kpi_card("Predicted attack", str(row.get("predicted_attack", "N/A")))
                with c2:
                    conf = row.get("confidence", np.nan)
                    kpi_card("Confidence", f"{conf:.1f}%" if pd.notna(conf) else "N/A")
                with c3:
                    risk = row.get("risk_score", np.nan)
                    cat = row.get("risk_category", "N/A")
                    kpi_card("Risk score", f"{risk:.1f}" if pd.notna(risk) else "N/A",
                             sub=cat, accent=RISK_COLORS.get(cat))

                if cls_art is not None:
                    top_feats = top_contributing_features(cls_art)
                    if top_feats is not None:
                        st.markdown("##### Top contributing features")
                        for feat, val in top_feats.items():
                            st.markdown(f"- `{feat}` (importance = {val:.3f})")

            st.markdown("##### Full results")
            st.dataframe(results, width='stretch', height=400)
            st.download_button("Download results as CSV", results.to_csv(index=False), "detection_results.csv")

# ===========================================================================
# PAGE: ANALYTICS
# ===========================================================================
elif page == "Analytics":
    st.markdown("## Model Performance Analytics")

    st.markdown("#### Classification model comparison")
    cls_table = load_comparison_table("classification")
    if cls_table is None:
        not_trained_message("02_Classification.ipynb")
    else:
        st.dataframe(cls_table, width='stretch')
        metric_col = "F1 (weighted)" if "F1 (weighted)" in cls_table.columns else cls_table.columns[1]
        fig = px.bar(cls_table.reset_index(), x="Model", y=metric_col, color=metric_col,
                     color_continuous_scale=["#1A2332", COLORS["accent"]])
        fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                           margin=dict(l=0, r=0, t=10, b=0), height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')
        if cls_art is not None and cls_art["info"]:
            st.caption(f"Currently deployed: **{cls_art['info'].get('best_model_name', 'unknown')}** "
                       f"(F1 weighted = {cls_art['info'].get('f1_weighted', float('nan')):.4f})")

    st.markdown("#### Regression model comparison")
    reg_table = load_comparison_table("regression")
    if reg_table is None:
        not_trained_message("03_Regression.ipynb")
    else:
        st.dataframe(reg_table, width='stretch')
        fig = px.bar(reg_table.reset_index(), x="Model", y="R2 Score", color="R2 Score",
                     color_continuous_scale=["#1A2332", COLORS["accent"]])
        fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                           margin=dict(l=0, r=0, t=10, b=0), height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')
        if reg_art is not None and reg_art["info"]:
            st.caption(f"Currently deployed: **{reg_art['info'].get('best_model_name', 'unknown')}** "
                       f"(R2 = {reg_art['info'].get('r2_score', float('nan')):.4f})")

# ===========================================================================
# PAGE: EXPLAINABILITY
# ===========================================================================
elif page == "Explainability":
    st.markdown("## Explainability")

    if cls_art is None:
        not_trained_message("02_Classification.ipynb")
    else:
        st.markdown("#### Global feature importance")
        top_feats = top_contributing_features(cls_art, n=15)
        if top_feats is not None:
            fig = px.bar(x=top_feats.values, y=top_feats.index, orientation="h",
                         color=top_feats.values, color_continuous_scale=["#1A2332", COLORS["accent"]])
            fig.update_layout(paper_bgcolor=COLORS["bg"], plot_bgcolor=COLORS["bg"], font_color=COLORS["text"],
                               margin=dict(l=0, r=0, t=10, b=0), height=420, coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"), xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig, width='stretch')
        else:
            st.info(f"**{cls_art['info'].get('best_model_name', 'The deployed model')}** does not expose "
                    "feature_importances_ or coefficients directly (expected for models like KNN, SVM with "
                    "a non-linear kernel, or Naive Bayes).")

        st.markdown("#### SHAP analysis")
        st.caption("Computed on demand from a small reference sample -- kept small deliberately, since exact "
                   "SHAP values are expensive even for tree models on large samples.")
        model_name = cls_art["info"].get("best_model_name", "")
        tree_based = any(k in model_name for k in ["Forest", "Tree", "Boosting", "Bagging", "AdaBoost"])
        if not tree_based:
            st.info(f"**{model_name}** is not tree-based, so exact SHAP is too expensive to compute here by "
                    "default. The global importance above is the practical alternative for this model type.")
        else:
            ref_df = load_reference_sample(max_rows=300)
            if ref_df is None:
                not_trained_message("01_EDA.ipynb")
            else:
                try:
                    import shap
                    label_col = "label" if "label" in ref_df.columns else ref_df.columns[-1]
                    X_ref = prep.standardize_columns(ref_df.drop(columns=[label_col]))
                    X_ref = X_ref.reindex(columns=cls_art["feature_names"], fill_value=0)
                    X_ref_scaled = cls_art["scaler"].transform(X_ref)

                    explainer = shap.TreeExplainer(cls_art["model"])
                    shap_values = explainer.shap_values(X_ref_scaled)
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(9, 6))
                    shap.summary_plot(shap_values, X_ref_scaled, feature_names=cls_art["feature_names"],
                                       show=False, max_display=10)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"SHAP could not be computed: {e}")

# ===========================================================================
# PAGE: THREAT INTELLIGENCE (clustering)
# ===========================================================================
elif page == "Threat Intelligence":
    st.markdown("## Threat Intelligence -- Behavioral Clusters")
    st.caption("Discovered without using attack labels during training. Labels are shown here only to "
               "interpret what each cluster turned out to represent.")

    if clu_art is None:
        not_trained_message("04_Clustering.ipynb")
    else:
        profile = clu_art["profile"]
        st.markdown("#### Cluster profile")
        cols = st.columns(len(profile))
        for col, (idx, row) in zip(cols, profile.iterrows()):
            with col:
                kpi_card(f"Cluster {idx}", row.get("dominant_label", "?"),
                         sub=f"{row.get('dominant_pct', 0):.1f}% | {int(row.get('cluster_size', 0)):,} rows")

        st.markdown("#### Cluster composition")
        st.dataframe(profile, width='stretch')

        if clu_art["dropped_features"]:
            st.caption(f"Correlation filtering removed {len(clu_art['dropped_features'])} redundant "
                       f"feature(s) before clustering: {', '.join(clu_art['dropped_features'])}")
