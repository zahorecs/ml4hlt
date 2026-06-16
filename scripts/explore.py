import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.loader import load_root, load_combined, summary
from src.config import PLOTS_DIR, IMAGE_WIDTH_NS


def plot_global_trigger_offset(slices: pd.DataFrame, out_dir: Path) -> None:
    trig = slices[slices["has_tpc_trigger"]]
    offsets_us = trig["tpc_trigger_offset"] / 1000.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(offsets_us, bins=100, range=(0, 500), color="steelblue", edgecolor="none")
    ax.set_xlabel("TPC trigger offset from slice start (µs)")
    ax.set_ylabel("Count")
    ax.set_title(f"TPC trigger arrival distribution  (n={len(trig)})")
    fig.tight_layout()
    fig.savefig(out_dir / "trigger_offset.pdf")
    plt.close(fig)
    print("Saved trigger_offset.pdf")


def plot_per_source_profiles(images: pd.DataFrame, out_dir: Path) -> None:
    source_ids = sorted(images["source_id"].unique())
    trig    = images[images["has_tpc_trigger"] == True]
    no_trig = images[images["has_tpc_trigger"] == False]

    n_cols = 3
    n_rows = int(np.ceil(len(source_ids) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             sharex=False)
    axes = axes.flatten()

    for idx, sid in enumerate(source_ids):
        ax = axes[idx]

        t_mean = (trig[trig["source_id"] == sid]
                  .groupby("image_index")["hit_count"].mean())
        n_mean = (no_trig[no_trig["source_id"] == sid]
                  .groupby("image_index")["hit_count"].mean())

        x = np.arange(50) * IMAGE_WIDTH_NS / 1000.0

        if not n_mean.empty:
            ax.plot(x, n_mean.reindex(range(50), fill_value=0),
                    color="steelblue", lw=1.2, label="no trigger")
        if not t_mean.empty:
            ax.plot(x, t_mean.reindex(range(50), fill_value=0),
                    color="crimson", lw=1.2, label="triggered")

        ax.set_title(f"source ID {sid}", fontsize=9)
        ax.set_xlabel("Image time (µs)", fontsize=7)
        ax.set_ylabel("Mean hits", fontsize=7)
        ax.tick_params(labelsize=7)
        tick_positions = x[::10]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{v:.3g}" for v in tick_positions], fontsize=7)

    for idx in range(len(source_ids), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9)
    fig.suptitle("Mean hit count per image — triggered vs non-triggered", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "per_source_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved per_source_profiles.pdf")


def plot_trigger_aligned_profiles(images: pd.DataFrame,
                                   slices: pd.DataFrame,
                                   out_dir: Path) -> None:
    trig_slices = slices[slices["has_tpc_trigger"]].copy()
    trig_slices["trigger_image"] = (
        trig_slices["tpc_trigger_offset"] / 10_000.0
    ).astype(int).clip(0, 49)

    merged = images.merge(
        trig_slices[["slice_number", "trigger_image"]],
        on="slice_number",
        how="inner"
    )
    merged["rel_image"] = merged["image_index"].astype(int) - merged["trigger_image"]

    source_ids = sorted(merged["source_id"].unique())
    n_cols = 3
    n_rows = int(np.ceil(len(source_ids) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             sharex=False)
    axes = axes.flatten()

    for idx, sid in enumerate(source_ids):
        ax = axes[idx]
        sub = merged[merged["source_id"] == sid]
        profile = sub.groupby("rel_image")["hit_count"].mean()
        x = profile.index.values * IMAGE_WIDTH_NS / 1000.0
        ax.plot(x, profile.values, color="darkorange", lw=1.2)
        ax.axvline(0, color="black", lw=0.8, ls="--", label="trigger")
        ax.set_title(f"source {sid}", fontsize=9)
        ax.set_xlabel("Time rel. to trigger (µs)", fontsize=7)
        ax.set_ylabel("Mean hits", fontsize=7)
        ax.tick_params(labelsize=7)
        tick_positions = x[::10]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{v:.3g}" for v in tick_positions], fontsize=7)

    for idx in range(len(source_ids), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Mean hit count aligned to TPC trigger", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "trigger_aligned_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved trigger_aligned_profiles.pdf")


def plot_feature_distributions(images: pd.DataFrame, out_dir: Path) -> None:
    features = ["hit_count", "active_channels", "active_ports",
                "port_entropy", "channel_spread", "hit_time_range"]
    features = [f for f in features if f in images.columns]

    trig    = images[images["has_tpc_trigger"] == True]
    no_trig = images[images["has_tpc_trigger"] == False]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        lo = images[feat].quantile(0.01)
        hi = images[feat].quantile(0.99)
        bins = np.linspace(lo, hi, 80)
        ax.hist(no_trig[feat], bins=bins, density=True,
                color="steelblue", alpha=0.6, label="no trigger")
        ax.hist(trig[feat],    bins=bins, density=True,
                color="crimson",   alpha=0.6, label="triggered")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature distributions: triggered vs non-triggered images")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.pdf")
    plt.close(fig)
    print("Saved feature_distributions.pdf")


def main():
    parser = argparse.ArgumentParser(
        description="Exploratory analysis of FeatureExtractor ROOT output.")
    parser.add_argument("input", help="Path to .features.root file or combined parquet directory.")
    parser.add_argument("--plots-dir", default=None,
                        help="Output directory for plots (default: plots/)")
    args = parser.parse_args()

    out_dir = Path(args.plots_dir) if args.plots_dir else PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(args.input)
    print(f"Loading {p} ...")
    if p.is_dir():
        slices, images = load_combined(
            p,
            slice_cols=["slice_number", "has_tpc_trigger", "tpc_trigger_offset"],
            image_cols=["slice_number", "source_id", "image_index", "hit_count",
                        "has_tpc_trigger", "tpc_trigger_offset",
                        "active_channels", "active_ports",
                        "port_entropy", "channel_spread", "hit_time_range"],
        )
    else:
        slices, images = load_root(p)
    summary(slices, images)

    print("\nGenerating plots...")
    plot_global_trigger_offset(slices, out_dir)
    plot_per_source_profiles(images, out_dir)
    plot_trigger_aligned_profiles(images, slices, out_dir)
    plot_feature_distributions(images, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()