"""Diagnostic and comparison plot generator for PathSnap candidate classification (Issue #5, Report §3.3).

Generates from actual experiment results:
1. `cv_macro_f1_comparison.png`: Cross-validation Macro F1-score (mean ± std) across all 5 models.
2. `student_test_top1_and_macro_f1.png`: Student-testing Top-1 Accuracy and Macro F1-score comparison.
3. `log_loss_comparison.png`: Cross-validation and Student-testing Log Loss comparison.
4. `inference_time_vs_accuracy.png`: Computational efficiency vs. Student-testing Top-1 Accuracy trade-off.
5. `confusion_matrices.png`: Student-testing confusion matrices for all 5 models.
6. `mlp_training_loss_curve.png`: Epoch-wise training loss and validation score history exclusively for
   `mlp_neural_net` (per Deliverables.pdf §3.3: no artificial training curves for non-iterative models).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_cv_macro_f1_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot 5-fold GroupKFold CV Macro F1-score (mean ± std) across all 5 models."""
    df = comparison_df.sort_values("cv_macro_f1_mean", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)

    y_pos = np.arange(len(df))
    means = df["cv_macro_f1_mean"].to_numpy(dtype=float)
    stds = df["cv_macro_f1_std"].to_numpy(dtype=float)
    labels = [f"{r['Model']}\n({r['Algorithm Family']})" for _, r in df.iterrows()]

    bars = ax.barh(
        y_pos,
        means,
        xerr=stds,
        capsize=4,
        color=PALETTE[: len(df)],
        alpha=0.88,
        edgecolor="#222222",
        height=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("5-Fold GroupKFold Macro F1-Score (Mean ± 1 SD)", fontsize=10, fontweight="bold")
    title = "Cross-Validation Macro F1-Score Comparison Across 5 Algorithm Families"
    if is_mock:
        title += " [MOCK HARNESS]"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(max(0.0, float(means.min()) - 0.08), min(1.03, float((means + stds).max()) + 0.04))
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            min(1.01, m + s + 0.004),
            bar.get_y() + bar.get_height() / 2.0,
            f"{m:.4f} ± {s:.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_student_test_top1_and_macro_f1(
    comparison_df: pd.DataFrame,
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot Student-Test Observation-level Top-1 Accuracy, Macro F1, and Route Continuity."""
    df = comparison_df.sort_values("student_test_top1_accuracy", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)

    x = np.arange(len(df))
    width = 0.26

    top1 = df["student_test_top1_accuracy"].to_numpy(dtype=float)
    macro_f1 = df["student_test_macro_f1"].to_numpy(dtype=float)
    continuity = df["route_continuity"].to_numpy(dtype=float)

    b1 = ax.bar(x - width, top1, width, label="Student-Test Top-1 Accuracy", color="#1f77b4", edgecolor="#222")
    b2 = ax.bar(x, macro_f1, width, label="Student-Test Macro F1-Score", color="#2ca02c", edgecolor="#222")
    b3 = ax.bar(x + width, continuity, width, label="Route Continuity (Classifier-Only)", color="#ff7f0e", edgecolor="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model_key"].tolist(), rotation=15, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Score [0 – 1]", fontsize=10, fontweight="bold")
    title = "Student-Testing Performance: Top-1 Accuracy, Macro F1 & Route Continuity"
    if is_mock:
        title += " [MOCK HARNESS]"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    low_bound = max(0.0, float(min(top1.min(), macro_f1.min(), continuity.min())) - 0.12)
    ax.set_ylim(low_bound, 1.06)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.008,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_log_loss_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot CV Log Loss vs. Student-Test Log Loss across all 5 models (lower is better)."""
    df = comparison_df.sort_values("student_test_log_loss", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=160)

    x = np.arange(len(df))
    width = 0.35

    cv_ll = df["cv_log_loss_mean"].to_numpy(dtype=float)
    st_ll = df["student_test_log_loss"].to_numpy(dtype=float)

    b1 = ax.bar(x - width / 2, cv_ll, width, label="5-Fold CV Log Loss", color="#4c78a8", edgecolor="#222")
    b2 = ax.bar(x + width / 2, st_ll, width, label="Student-Test Log Loss", color="#e45756", edgecolor="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(df["model_key"].tolist(), rotation=15, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Log Loss (Lower = Better Calibration)", fontsize=10, fontweight="bold")
    title = "Probability Calibration Quality: Log Loss Comparison Across Models"
    if is_mock:
        title += " [MOCK HARNESS]"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    max_ll = float(max(cv_ll.max(), st_ll.max()))
    ax.set_ylim(0.0, max_ll * 1.22 + 0.01)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + max_ll * 0.02,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_inference_time_vs_accuracy(
    comparison_df: pd.DataFrame,
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot Inference Latency (ms/trip) vs. Student-Test Top-1 Accuracy to justify model selection."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)

    x_ms = comparison_df["infer_ms_per_trip"].to_numpy(dtype=float)
    y_top1 = comparison_df["student_test_top1_accuracy"].to_numpy(dtype=float)
    is_best = comparison_df["selected_final_model"].to_numpy(dtype=bool)

    for idx, row in comparison_df.iterrows():
        color = PALETTE[idx % len(PALETTE)]
        marker = "*" if bool(row["selected_final_model"]) else "o"
        size = 240 if bool(row["selected_final_model"]) else 120
        ax.scatter(
            float(row["infer_ms_per_trip"]),
            float(row["student_test_top1_accuracy"]),
            s=size,
            color=color,
            marker=marker,
            edgecolors="#111111",
            linewidths=1.2,
            label=f"{row['model_key']}" + (" (Selected)" if bool(row["selected_final_model"]) else ""),
            zorder=4,
        )
        ax.annotate(
            str(row["model_key"]),
            (float(row["infer_ms_per_trip"]), float(row["student_test_top1_accuracy"])),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xlabel("Student-Test Inference Time per Trip (ms, Lower = Faster)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Student-Test Observation Top-1 Accuracy", fontsize=10, fontweight="bold")
    title = "Model Selection Trade-Off: Inference Speed vs. Top-1 Accuracy"
    if is_mock:
        title += " [MOCK HARNESS]"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8.5, frameon=True)

    if len(y_top1) > 0 and np.any(is_best):
        pass

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrices(
    confusion_records: Mapping[str, Mapping[str, int]],
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot a grid of 2x2 Student-Test confusion matrices across all 5 models."""
    keys = list(confusion_records.keys())
    n_models = len(keys)
    fig, axes = plt.subplots(1, n_models, figsize=(3.4 * n_models, 3.6), dpi=160)
    if n_models == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        rec = confusion_records[key]
        mat = np.array(
            [
                [rec["tn"], rec["fp"]],
                [rec["fn"], rec["tp"]],
            ],
            dtype=int,
        )
        row_sums = np.maximum(mat.sum(axis=1, keepdims=True), 1)
        norm_mat = mat / row_sums

        ax.imshow(norm_mat, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(key, fontsize=9.5, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8.5)
        ax.set_yticklabels(["True 0", "True 1"], fontsize=8.5)

        for i in range(2):
            for j in range(2):
                val = mat[i, j]
                pct = norm_mat[i, j] * 100.0
                text_color = "white" if norm_mat[i, j] > 0.55 else "#111111"
                ax.text(
                    j,
                    i,
                    f"{val}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="bold",
                )

    suptitle = "Student-Testing Candidate Classification Confusion Matrices"
    if is_mock:
        suptitle += " [MOCK HARNESS]"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_mlp_training_history(
    loss_curve: List[float],
    val_scores: Optional[List[float]],
    output_path: Path,
    is_mock: bool = False,
) -> Path:
    """Plot epoch-wise training loss and validation score exclusively for `mlp_neural_net`.

    Per Deliverables.pdf §3.3:
    'Training and validation loss plots only for models that produce epoch-wise training histories.
     Do not create artificial training curves for algorithms that do not produce training histories.'
    """
    epochs = np.arange(1, len(loss_curve) + 1)
    fig, ax1 = plt.subplots(figsize=(7.6, 4.4), dpi=160)

    ax1.plot(epochs, loss_curve, color="#d62728", linewidth=2.0, marker="o", markersize=3.5, label="Training Cross-Entropy Loss")
    ax1.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Training Cross-Entropy Loss", color="#d62728", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle="--", alpha=0.4)

    if val_scores:
        ax2 = ax1.twinx()
        val_ep = np.arange(1, len(val_scores) + 1)
        ax2.plot(
            val_ep,
            val_scores,
            color="#1f77b4",
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=3.5,
            label="Internal Validation Accuracy",
        )
        ax2.set_ylabel("Hold-Out Validation Accuracy", color="#1f77b4", fontsize=10, fontweight="bold")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")

    title = "Shallow MLP Neural Network (64-32): Epoch-Wise Training & Validation History"
    if is_mock:
        title += " [MOCK HARNESS]"
    ax1.set_title(title, fontsize=10.5, fontweight="bold", pad=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_comparison_plots(
    comparison_df: pd.DataFrame,
    confusion_records: Mapping[str, Mapping[str, int]],
    mlp_history: Optional[Dict[str, Any]] = None,
    figures_dir: Path = FIGURES_DIR,
    is_mock: bool = False,
) -> Dict[str, Path]:
    """Generate all compulsory and diagnostic model evaluation charts in `figures_dir`."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Path] = {}

    generated["cv_macro_f1"] = plot_cv_macro_f1_comparison(
        comparison_df, figures_dir / "cv_macro_f1_comparison.png", is_mock=is_mock
    )
    generated["student_test_top1_macro_f1"] = plot_student_test_top1_and_macro_f1(
        comparison_df, figures_dir / "student_test_top1_and_macro_f1.png", is_mock=is_mock
    )
    generated["log_loss"] = plot_log_loss_comparison(
        comparison_df, figures_dir / "log_loss_comparison.png", is_mock=is_mock
    )
    generated["inference_vs_accuracy"] = plot_inference_time_vs_accuracy(
        comparison_df, figures_dir / "inference_time_vs_accuracy.png", is_mock=is_mock
    )
    generated["confusion_matrices"] = plot_confusion_matrices(
        confusion_records, figures_dir / "confusion_matrices.png", is_mock=is_mock
    )
    if mlp_history and mlp_history.get("loss_curve"):
        generated["mlp_training_loss_curve"] = plot_mlp_training_history(
            loss_curve=list(mlp_history["loss_curve"]),
            val_scores=list(mlp_history.get("validation_scores") or []),
            output_path=figures_dir / "mlp_training_loss_curve.png",
            is_mock=is_mock,
        )

    return generated
