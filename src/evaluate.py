from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def compute_metrics(
    y_true,
    y_pred,
    model_name: str = "model",
) -> Dict[str, float]:
    """Compute accuracy, macro precision/recall/F1 and return as a dict.

    Parameters
    ----------
    y_true : array-like
        Ground-truth category labels.
    y_pred : array-like
        Predicted category labels from the model.
    model_name : str
        Label used as the ``model`` key in the returned dict.

    Returns
    -------
    dict with keys: model, accuracy, precision_macro, recall_macro, f1_macro
    """
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 4
        ),
    }


def print_report(metrics: Dict[str, float]) -> None:
    """Pretty-print the metrics dict returned by compute_metrics."""
    model = metrics.get("model", "unknown")
    print(f"\n{'=' * 50}")
    print(f"  Results: {model}")
    print(f"{'=' * 50}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro) : {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)    : {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro)        : {metrics['f1_macro']:.4f}")
    print(f"{'=' * 50}\n")


def print_classification_report(y_true, y_pred) -> None:
    """Print sklearn's per-class precision / recall / F1 table."""
    print(classification_report(y_true, y_pred, zero_division=0))


def save_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "model",
    output_dir: Path = Path("models"),
) -> None:
    """Save a confusion matrix heatmap as a PNG file.

    Requires matplotlib. Silently skips if matplotlib is not installed.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Used in the plot title and output filename.
    output_dir : Path
        Directory where the PNG is saved.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed — skipping confusion matrix plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    out_path = output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")


def save_learning_curve(
    estimator,
    model_name: str = "model",
    output_dir: Path = Path("models"),
) -> None:
    """Save a learning-curve PNG: training loss + validation accuracy per epoch.

    Works with any estimator exposing ``loss_curve_`` (and optionally
    ``validation_scores_``) -- e.g. ``MLPClassifier`` fit with
    ``early_stopping=True``. Silently skips if matplotlib isn't installed
    or the estimator hasn't recorded a loss curve.
    """
    if not _MATPLOTLIB_AVAILABLE:
        print("matplotlib not installed -- skipping learning curve plot.")
        return

    loss = getattr(estimator, "loss_curve_", None)
    val_scores = getattr(estimator, "validation_scores_", None)
    if not loss:
        print(f"{model_name} has no loss_curve_ -- skipping learning curve plot.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(range(1, len(loss) + 1), loss, color="tab:blue", label="Training loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss", color="tab:blue")
    ax_loss.tick_params(axis="y", labelcolor="tab:blue")
    ax_loss.grid(alpha=0.3)

    if val_scores:
        ax_acc = ax_loss.twinx()
        ax_acc.plot(
            range(1, len(val_scores) + 1),
            val_scores,
            color="tab:orange",
            label="Val accuracy (early-stop holdout)",
        )
        ax_acc.set_ylabel("Validation accuracy", color="tab:orange")
        ax_acc.tick_params(axis="y", labelcolor="tab:orange")

    fig.suptitle(f"Learning Curve -- {model_name}", fontsize=14)
    fig.tight_layout()
    out_path = output_dir / f"{model_name.lower().replace(' ', '_')}_learning_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Learning curve saved to {out_path}")


def print_runtime_summary(search_time: float, refit_time: float) -> None:
    """Print a runtime block for the rubric's Runtime Analysis item."""
    total = search_time + refit_time
    print("\n" + "=" * 50)
    print("  Runtime")
    print("=" * 50)
    print(f"  Search time : {search_time:7.1f}s")
    print(f"  Refit time  : {refit_time:7.1f}s")
    print(f"  Total       : {total:7.1f}s")
    print("=" * 50)
