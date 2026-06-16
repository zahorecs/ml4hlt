import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = np.linspace(0, 1, 500)
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapz(tpr[::-1], fpr[::-1]))


def plot_roc(y_true: np.ndarray, y_score: np.ndarray,
             out_dir: Path, label: str = "Bayesian Gaussian") -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    area = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{label}  (AUC = {area:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — timeslice classification")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.pdf")
    plt.close(fig)
    print(f"Saved roc_curve.pdf  (AUC = {area:.3f})")
    return area


def plot_score_distribution(y_true: np.ndarray, y_score: np.ndarray,
                             out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 60)
    ax.hist(y_score[y_true == 0], bins=bins, density=True,
            color="steelblue", alpha=0.6, label="no trigger")
    ax.hist(y_score[y_true == 1], bins=bins, density=True,
            color="crimson",   alpha=0.6, label="triggered")
    ax.set_xlabel("P(signal | data)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior signal probability distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "score_distribution.pdf")
    plt.close(fig)
    print("Saved score_distribution.pdf")


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray,
                     out_dir: Path, n_bins: int = 10) -> None:
    bins   = np.linspace(0, 1, n_bins + 1)
    centres, frac_pos = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        centres.append((lo + hi) / 2)
        frac_pos.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(centres, frac_pos, "o-", color="darkorange", label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.pdf")
    plt.close(fig)
    print("Saved calibration.pdf")


def print_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = 0.5) -> None:
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    print(f"\nConfusion matrix (threshold={threshold:.2f})")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision : {tp/(tp+fp):.3f}" if (tp+fp) > 0 else "  Precision : N/A")
    print(f"  Recall    : {tp/(tp+fn):.3f}" if (tp+fn) > 0 else "  Recall    : N/A")
    print(f"  Specificity: {tn/(tn+fp):.3f}" if (tn+fp) > 0 else "  Specificity: N/A")