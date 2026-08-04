"""
utils.py
========
Shared helper functions used across the EDA, classification, regression,
and clustering notebooks for the CyberThreat-ML project.

Keeping these in one place means every notebook plots confusion matrices,
ROC curves, and comparison tables the same way, and every model is timed
and saved the same way. This directly supports the "avoid repeated code"
requirement instead of re-writing plotting logic in each notebook.
"""

import os
import time
from contextlib import contextmanager

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.preprocessing import label_binarize


# ---------------------------------------------------------------------------
# Consistent visual style across all notebooks
# ---------------------------------------------------------------------------
def set_plot_style():
    """Apply one consistent matplotlib/seaborn style for every notebook."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["font.size"] = 10


@contextmanager
def timer(label="Task"):
    """Context manager that prints how long a block of code took.
    Used to report per-model training time -- relevant when discussing
    computational efficiency / hardware feasibility in the report."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[{label}] completed in {elapsed:.2f} seconds")


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------
def evaluate_classifier(model, X_test, y_test, model_name, classes, y_proba=None):
    """
    Compute the mandatory classification metrics for one trained model.

    classes must match the column order of predict_proba (i.e. pass
    model.classes_) so the one-vs-rest ROC-AUC lines up correctly.
    """
    y_pred = model.predict(X_test)

    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 (macro)": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if y_proba is None and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    if y_proba is not None:
        try:
            y_test_bin = label_binarize(y_test, classes=classes)
            result["ROC-AUC (OvR weighted)"] = roc_auc_score(
                y_test_bin, y_proba, average="weighted", multi_class="ovr"
            )
        except Exception:
            result["ROC-AUC (OvR weighted)"] = np.nan
    else:
        result["ROC-AUC (OvR weighted)"] = np.nan

    result["_y_pred"] = y_pred
    result["_y_proba"] = y_proba
    return result


def plot_confusion_matrix(y_test, y_pred, classes, model_name, save_path=None, normalize=True):
    """Plot (and optionally save) a labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    if normalize:
        with np.errstate(all="ignore"):
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, cbar=True)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix -- {model_name}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return cm


def plot_multiclass_roc(y_test, y_proba, classes, model_name, save_path=None):
    """Plot one-vs-rest ROC curves for every class present in y_test."""
    y_test_bin = label_binarize(y_test, classes=classes)
    if y_test_bin.shape[1] == 1:  # binary edge case from label_binarize
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        if y_test_bin[:, i].sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in fpr:
        ax.plot(fpr[i], tpr[i], lw=1.5, label=f"{classes[i]} (AUC = {roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC curves -- {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return roc_auc


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------
def evaluate_regressor(model, X_test, y_test, model_name):
    """Compute the mandatory regression metrics for one trained model."""
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {
        "Model": model_name,
        "R2 Score": r2_score(y_test, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_test, y_pred),
        "_y_pred": y_pred,
    }


def plot_actual_vs_predicted(y_test, y_pred, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, edgecolor="none")
    lo = min(np.min(y_test), np.min(y_pred))
    hi = max(np.max(y_test), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect prediction")
    ax.set_xlabel("Actual risk score")
    ax.set_ylabel("Predicted risk score")
    ax.set_title(f"Actual vs predicted -- {model_name}")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_residuals(y_test, y_pred, model_name, save_path=None):
    residuals = np.asarray(y_test) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, edgecolor="none")
    axes[0].axhline(0, color="red", linestyle="--", lw=2)
    axes[0].set_xlabel("Predicted risk score")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predicted")

    sns.histplot(residuals, kde=True, ax=axes[1], bins=40)
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual")

    fig.suptitle(f"Residual analysis -- {model_name}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return residuals


# ---------------------------------------------------------------------------
# Clustering visualization
# ---------------------------------------------------------------------------
def plot_elbow_curve(k_values, inertias, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_values), inertias, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow method for optimal k")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_2d_clusters(X_2d, labels, title, save_path=None, palette="tab10"):
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=palette, s=8, alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
def save_model(model, name, models_dir="../models"):
    """Save a fitted model (or any picklable object, e.g. a fitted scaler)."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved: {path}")
    return path


def load_model(name, models_dir="../models"):
    path = os.path.join(models_dir, f"{name}.joblib")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Comparison table helper
# ---------------------------------------------------------------------------
def build_comparison_table(results_list, primary_metric):
    """
    Turn a list of evaluate_classifier()/evaluate_regressor() result dicts
    into a single, ranked comparison DataFrame (drops internal _y_pred /
    _y_proba keys used only for plotting).
    """
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results_list]
    df = pd.DataFrame(rows).set_index("Model")
    ascending = primary_metric in ("RMSE", "MAE")  # lower is better for these
    df = df.sort_values(by=primary_metric, ascending=ascending)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df
