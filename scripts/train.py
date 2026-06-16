import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.loader import load_root, load_combined, summary
from src.features.builder import build_feature_matrix, get_feature_columns, split_features_labels
from src.models.gaussian_classifier import BayesianGaussianClassifier
from src.utils.preprocessing import StandardScaler, PCA
from src.utils.evaluation import (
    plot_roc, plot_score_distribution,
    plot_calibration, print_confusion
)
from src.config import PLOTS_DIR


def run(root_file: str, plots_dir: Path, pca_threshold: float,
        prior_strength: float, shared_cov: bool) -> None:

    p = Path(root_file)
    print(f"Loading {p} ...")
    if p.is_dir():
        slices, images = load_combined(p)
    else:
        slices, images = load_root(p)
    summary(slices, images)

    print("\nBuilding feature matrix ...")
    df = build_feature_matrix(slices, images)
    feature_cols = get_feature_columns(df)
    print(f"Feature matrix: {len(df)} timeslices x {len(feature_cols)} features")

    X, y = split_features_labels(df)

    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.sum() > 0:
        print(f"Dropping {nan_mask.sum()} rows with NaN features")
        X = X[~nan_mask]
        y = y[~nan_mask]

    print(f"Class balance: {y.sum()} signal ({100*y.mean():.1f}%)  "
          f"{(1-y).sum()} background")

    print(f"\nPre-processing ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(variance_threshold=pca_threshold)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {X_pca.shape[1]} components explain "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% of variance")

    print(f"\nFitting Bayesian Gaussian classifier ...")
    print(f"  prior_strength={prior_strength}  "
          f"shared_covariance={shared_cov}")

    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_true = []
    all_prob = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_pca, y)):
        X_tr, X_val = X_pca[train_idx], X_pca[val_idx]
        y_tr, y_val = y[train_idx],     y[val_idx]

        clf = BayesianGaussianClassifier(
            prior_strength=prior_strength,
            shared_covariance=shared_cov,
        )
        clf.fit(X_tr, y_tr)

        probs = clf.predict_signal_proba(X_val)
        all_true.append(y_val)
        all_prob.append(probs)

        fold_auc = _quick_auc(y_val, probs)
        print(f"  Fold {fold+1}: AUC = {fold_auc:.3f}  "
              f"(n_val={len(y_val)}, "
              f"signal={y_val.sum()})")

    y_true_all = np.concatenate(all_true)
    y_prob_all = np.concatenate(all_prob)

    print("\nGenerating evaluation plots ...")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_roc(y_true_all, y_prob_all, plots_dir)
    plot_score_distribution(y_true_all, y_prob_all, plots_dir)
    plot_calibration(y_true_all, y_prob_all, plots_dir)

    y_pred_all = (y_prob_all >= 0.5).astype(int)
    print_confusion(y_true_all, y_pred_all, threshold=0.5)

    print(f"\nAll plots saved to {plots_dir}")


def _quick_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order  = np.argsort(y_score)[::-1]
    y_sort = y_true[order]
    n_pos  = y_true.sum()
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, fp = 0, 0
    auc    = 0.0
    prev_fp = 0
    for label in y_sort:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Bayesian Gaussian classifier on timeslice features.")
    parser.add_argument("root_file", help="Path to .features.root file")
    parser.add_argument("--plots-dir",      default=None)
    parser.add_argument("--pca-threshold",  type=float, default=0.95,
                        help="Fraction of variance to retain in PCA (default 0.95)")
    parser.add_argument("--prior-strength", type=float, default=1.0,
                        help="Prior strength kappa_0 (default 1.0)")
    parser.add_argument("--no-shared-cov",  action="store_true",
                        help="Use separate covariance per class (QDA instead of LDA)")
    args = parser.parse_args()

    out_dir = Path(args.plots_dir) if args.plots_dir else PLOTS_DIR

    run(
        root_file      = args.root_file,
        plots_dir      = out_dir,
        pca_threshold  = args.pca_threshold,
        prior_strength = args.prior_strength,
        shared_cov     = not args.no_shared_cov,
    )


if __name__ == "__main__":
    main()