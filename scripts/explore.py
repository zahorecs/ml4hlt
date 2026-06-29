"""
explore.py  —  memory-efficient exploratory analysis
=====================================================
Streams images.parquet in batches instead of loading it all at once.
All aggregations needed for the four plots are accumulated in a single
pass, so peak RAM usage is proportional to (n_sources × n_images × 2)
— a few thousand numbers regardless of dataset size.

Usage
-----
  python explore.py <input>           # .features.root file  OR
  python explore.py <combined_dir>    # directory with slices/images .parquet
  python explore.py <combined_dir> --plots-dir plots/
"""

import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.loader import load_root, summary
from src.config import PLOTS_DIR, IMAGE_WIDTH_NS

BATCH_SIZE = 200_000   # rows per parquet batch — tune to your RAM


# ---------------------------------------------------------------------------
# Streaming aggregation
# ---------------------------------------------------------------------------

def _stream_aggregate(images_path: Path,
                      triggered_slices: set,
                      trigger_image_map: dict) -> dict:
    """
    Single pass over images.parquet using pure Arrow→numpy conversions
    (no to_pylist), bincount instead of np.add.at, and no per-row Python loops.
    """
    import pyarrow.compute as pc
    import time

    needed_cols = [
        "slice_number", "source_id", "image_index", "hit_count",
        "has_tpc_trigger", "active_channels", "active_ports",
        "port_entropy", "channel_spread", "hit_time_range",
    ]

    REL_RANGE = 50
    N_REL     = 2 * REL_RANGE + 1
    feat_cols = ["hit_count", "active_channels", "active_ports",
                 "port_entropy", "channel_spread", "hit_time_range"]

    # --- Pre-build source-id → compact index mapping -------------------
    # We don't know source IDs upfront, so we grow these lazily and keep
    # a sid→idx dict.  All accumulators are 2-D arrays: [n_sources, bins].
    sid_to_idx: dict[int, int] = {}
    # prof arrays grown on demand: list of np.array rows
    prof_sum_t  = []   # list of np.zeros(50) — one per source, triggered
    prof_cnt_t  = []
    prof_sum_n  = []   # non-triggered
    prof_cnt_n  = []
    align_sum_a = []   # list of np.zeros(N_REL)
    align_cnt_a = []

    def _get_idx(sid: int) -> int:
        if sid not in sid_to_idx:
            idx = len(sid_to_idx)
            sid_to_idx[sid] = idx
            prof_sum_t.append(np.zeros(50,    dtype=np.float64))
            prof_cnt_t.append(np.zeros(50,    dtype=np.float64))
            prof_sum_n.append(np.zeros(50,    dtype=np.float64))
            prof_cnt_n.append(np.zeros(50,    dtype=np.float64))
            align_sum_a.append(np.zeros(N_REL, dtype=np.float64))
            align_cnt_a.append(np.zeros(N_REL, dtype=np.float64))
        return sid_to_idx[sid]

    # Fixed-size reservoir arrays (float32 to halve memory).
    RESERVOIR = 300_000
    feat_res = {f: {"trig":    np.empty(RESERVOIR, dtype=np.float32),
                    "no_trig": np.empty(RESERVOIR, dtype=np.float32)}
                for f in feat_cols}
    feat_n   = {f: {"trig": 0, "no_trig": 0} for f in feat_cols}

    # Lookup array: slice_number → trigger_image index (-1 = not triggered)
    max_slice    = max(trigger_image_map.keys()) + 1 if trigger_image_map else 0
    trig_img_arr = np.full(max_slice, -1, dtype=np.int32)
    for sn, ti in trigger_image_map.items():
        trig_img_arr[sn] = ti

    dataset    = pq.ParquetFile(images_path)
    total_rows = dataset.metadata.num_rows
    rows_done  = 0
    batch_num  = 0
    t0         = time.time()

    for batch in dataset.iter_batches(batch_size=BATCH_SIZE, columns=needed_cols):
        rows_done += batch.num_rows
        batch_num += 1

        elapsed = time.time() - t0
        pct     = 100.0 * rows_done / total_rows
        rate    = rows_done / elapsed if elapsed > 0 else 0
        eta_s   = (total_rows - rows_done) / rate if rate > 0 else 0
        print(f"  batch {batch_num:4d} | {rows_done:>12,} / {total_rows:,} rows "
              f"({pct:5.1f}%) | {rate/1e6:.2f}M rows/s | ETA {eta_s/60:.1f} min",
              flush=True)

        # --- Arrow → numpy in one shot (zero-copy where possible) -------
        sid_np  = batch.column("source_id").to_pydict() if False else                   batch.column("source_id").to_numpy(zero_copy_only=False).astype(np.int32)
        img_np  = batch.column("image_index").to_numpy(zero_copy_only=False).astype(np.int32)
        hc_np   = batch.column("hit_count").to_numpy(zero_copy_only=False).astype(np.float64)
        sn_np   = batch.column("slice_number").to_numpy(zero_copy_only=False).astype(np.int32)
        trig_np = batch.column("has_tpc_trigger").to_numpy(zero_copy_only=False).astype(bool)

        valid_img = (img_np >= 0) & (img_np < 50)

        # --- Ensure all source IDs in this batch have accumulators ------
        for sid in np.unique(sid_np):
            _get_idx(int(sid))

        # Build a compact "source index" array for the whole batch
        # (vectorised dict lookup via a temporary dense array)
        sid_max  = int(sid_np.max()) + 1
        dense    = np.full(sid_max, -1, dtype=np.int32)
        for sid, idx in sid_to_idx.items():
            if sid < sid_max:
                dense[sid] = idx
        sidx_np = dense[sid_np]   # compact source index per row

        # ---- per-source profiles via bincount --------------------------
        # Encode (source_idx, image_index) → single integer key
        # key = sidx * 50 + img_idx  (both small)
        n_src = len(sid_to_idx)
        for label, mask in [("trig", trig_np), ("no_trig", ~trig_np)]:
            m = mask & valid_img
            if not m.any():
                continue
            keys = sidx_np[m] * 50 + img_np[m]   # shape (m.sum(),)
            n_bins = n_src * 50

            hc_sum = np.bincount(keys, weights=hc_np[m],  minlength=n_bins)
            hc_cnt = np.bincount(keys,                     minlength=n_bins).astype(np.float64)

            # Scatter back into per-source arrays
            hc_sum_2d = hc_sum.reshape(n_src, 50)
            hc_cnt_2d = hc_cnt.reshape(n_src, 50)

            if label == "trig":
                for idx in range(n_src):
                    prof_sum_t[idx] += hc_sum_2d[idx]
                    prof_cnt_t[idx] += hc_cnt_2d[idx]
            else:
                for idx in range(n_src):
                    prof_sum_n[idx] += hc_sum_2d[idx]
                    prof_cnt_n[idx] += hc_cnt_2d[idx]

        # ---- trigger-aligned profile via bincount ----------------------
        if max_slice > 0:
            safe_sn    = np.clip(sn_np, 0, max_slice - 1)
            ti_np      = trig_img_arr[safe_sn]
            valid_trig = (sn_np < max_slice) & (ti_np >= 0)
            if valid_trig.any():
                rel    = img_np[valid_trig] - ti_np[valid_trig]
                r_idx  = rel + REL_RANGE
                in_rng = (r_idx >= 0) & (r_idx < N_REL)
                s_t    = sidx_np[valid_trig][in_rng]
                r_t    = r_idx[in_rng]
                hc_t   = hc_np[valid_trig][in_rng]

                keys_a   = s_t * N_REL + r_t
                n_bins_a = n_src * N_REL
                a_sum = np.bincount(keys_a, weights=hc_t, minlength=n_bins_a).reshape(n_src, N_REL)
                a_cnt = np.bincount(keys_a,               minlength=n_bins_a).reshape(n_src, N_REL).astype(np.float64)
                for idx in range(n_src):
                    align_sum_a[idx] += a_sum[idx]
                    align_cnt_a[idx] += a_cnt[idx]

        # ---- feature reservoir (Vitter R, zero to_pylist) --------------
        for feat in feat_cols:
            col = batch.column(feat)
            vals_np = col.to_numpy(zero_copy_only=False).astype(np.float32)
            for label, mask in [("trig", trig_np), ("no_trig", ~trig_np)]:
                v = vals_np[mask]
                if len(v) == 0:
                    continue
                n     = feat_n[feat][label]
                res   = feat_res[feat][label]
                space = RESERVOIR - n
                if space >= len(v):
                    res[n:n + len(v)] = v
                    feat_n[feat][label] += len(v)
                else:
                    if space > 0:
                        res[n:] = v[:space]
                        feat_n[feat][label] = RESERVOIR
                        v = v[space:]
                    total_seen  = n + len(v)
                    replace_idx = np.random.randint(0, total_seen, size=len(v))
                    keep        = replace_idx < RESERVOIR
                    res[replace_idx[keep]] = v[keep]

    # --- Reassemble dict-keyed outputs expected by plot functions -------
    prof_sum  = {}
    prof_cnt  = {}
    align_sum = {}
    align_cnt = {}
    for sid, idx in sid_to_idx.items():
        prof_sum[sid]  = {"trig": prof_sum_t[idx], "no_trig": prof_sum_n[idx]}
        prof_cnt[sid]  = {"trig": prof_cnt_t[idx], "no_trig": prof_cnt_n[idx]}
        align_sum[sid] = align_sum_a[idx]
        align_cnt[sid] = align_cnt_a[idx]

    feat_vals = {
        f: {label: feat_res[f][label][:feat_n[f][label]] for label in ("trig", "no_trig")}
        for f in feat_cols
    }

    return {
        "prof_sum":  prof_sum,
        "prof_cnt":  prof_cnt,
        "align_sum": align_sum,
        "align_cnt": align_cnt,
        "feat_vals": feat_vals,
        "REL_RANGE": REL_RANGE,
    }


# ---------------------------------------------------------------------------
# Plot functions (work on aggregated summaries, not raw data)
# ---------------------------------------------------------------------------

def plot_global_trigger_offset(slices: pd.DataFrame, out_dir: Path) -> None:
    trig = slices[slices["has_tpc_trigger"]]
    offsets_us = trig["tpc_trigger_offset"] / 1000.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(offsets_us, bins=100, range=(0, 500),
            color="steelblue", edgecolor="none")
    ax.set_xlabel("TPC trigger offset from slice start (µs)")
    ax.set_ylabel("Count")
    ax.set_title(f"TPC trigger arrival distribution  (n={len(trig)})")
    fig.tight_layout()
    fig.savefig(out_dir / "trigger_offset.pdf")
    plt.close(fig)
    print("Saved trigger_offset.pdf")


def plot_per_source_profiles(agg: dict, out_dir: Path) -> None:
    prof_sum = agg["prof_sum"]
    prof_cnt = agg["prof_cnt"]
    source_ids = sorted(prof_sum.keys())

    n_cols = 3
    n_rows = max(1, int(np.ceil(len(source_ids) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             sharex=False)
    axes = np.array(axes).flatten()

    x = np.arange(50) * IMAGE_WIDTH_NS / 1000.0

    for idx, sid in enumerate(source_ids):
        ax = axes[idx]
        for label, color in [("no_trig", "steelblue"), ("trig", "crimson")]:
            cnt = prof_cnt[sid][label]
            s   = prof_sum[sid][label]
            mean = np.where(cnt > 0, s / np.where(cnt > 0, cnt, 1.0), 0.0)
            ax.plot(x, mean, color=color, lw=1.2,
                    label="triggered" if label == "trig" else "no trigger")

        ax.set_title(f"source ID {sid}", fontsize=9)
        ax.set_xlabel("Image time (µs)", fontsize=7)
        ax.set_ylabel("Mean hits", fontsize=7)
        ax.tick_params(labelsize=7)
        tick_positions = x[::10]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{v:.3g}" for v in tick_positions], fontsize=7)

    for idx in range(len(source_ids), len(axes)):
        axes[idx].set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower right", fontsize=9)
    fig.suptitle("Mean hit count per image — triggered vs non-triggered", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "per_source_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved per_source_profiles.pdf")


def plot_trigger_aligned_profiles(agg: dict, out_dir: Path) -> None:
    align_sum  = agg["align_sum"]
    align_cnt  = agg["align_cnt"]
    REL_RANGE  = agg["REL_RANGE"]
    source_ids = sorted(align_sum.keys())

    n_cols = 3
    n_rows = max(1, int(np.ceil(len(source_ids) / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3 * n_rows),
                             sharex=False)
    axes = np.array(axes).flatten()

    rel_indices = np.arange(-REL_RANGE, REL_RANGE + 1)
    x_us = rel_indices * IMAGE_WIDTH_NS / 1000.0

    for idx, sid in enumerate(source_ids):
        ax = axes[idx]
        cnt  = align_cnt[sid]
        s    = align_sum[sid]
        mean = np.where(cnt > 0, s / np.where(cnt > 0, cnt, 1.0), np.nan)
        ax.plot(x_us, mean, color="darkorange", lw=1.2)
        ax.axvline(0, color="black", lw=0.8, ls="--", label="trigger")
        ax.set_title(f"source {sid}", fontsize=9)
        ax.set_xlabel("Time rel. to trigger (µs)", fontsize=7)
        ax.set_ylabel("Mean hits", fontsize=7)
        ax.tick_params(labelsize=7)

    for idx in range(len(source_ids), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Mean hit count aligned to TPC trigger", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "trigger_aligned_profiles.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved trigger_aligned_profiles.pdf")


def plot_feature_distributions(agg: dict, out_dir: Path) -> None:
    feat_vals = agg["feat_vals"]
    feat_cols = [f for f in feat_vals if
                 len(feat_vals[f]["trig"]) > 0 or len(feat_vals[f]["no_trig"]) > 0]

    n_feats = len(feat_cols)
    n_cols  = 3
    n_rows  = max(1, int(np.ceil(n_feats / n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(feat_cols):
        ax   = axes[idx]
        trig    = np.array(feat_vals[feat]["trig"],    dtype=float)
        no_trig = np.array(feat_vals[feat]["no_trig"], dtype=float)
        all_v   = np.concatenate([trig, no_trig]) if len(trig) and len(no_trig) \
                  else (trig if len(trig) else no_trig)
        lo = np.nanpercentile(all_v, 1)
        hi = np.nanpercentile(all_v, 99)
        bins = np.linspace(lo, hi, 80)
        if len(no_trig):
            ax.hist(no_trig, bins=bins, density=True,
                    color="steelblue", alpha=0.6, label="no trigger")
        if len(trig):
            ax.hist(trig, bins=bins, density=True,
                    color="crimson", alpha=0.6, label="triggered")
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for idx in range(n_feats, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Feature distributions: triggered vs non-triggered images")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_distributions.pdf")
    plt.close(fig)
    print("Saved feature_distributions.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient exploratory analysis of FeatureExtractor output.")
    parser.add_argument("input",
                        help="Path to .features.root file OR combined parquet directory.")
    parser.add_argument("--plots-dir", default=None,
                        help="Output directory for plots (default: plots/)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Rows per parquet batch (default {BATCH_SIZE})")
    args = parser.parse_args()

    out_dir = Path(args.plots_dir) if args.plots_dir else PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(args.input)
    print(f"Loading {p} ...")

    if not p.is_dir():
        # ---- single ROOT file: load everything (usually small) ----------
        slices, images = load_root(p)
        summary(slices, images)
        print("\nGenerating plots (in-memory mode for single ROOT file)...")

        triggered_slices   = set(slices.loc[slices["has_tpc_trigger"], "slice_number"])
        trig_slices_df     = slices[slices["has_tpc_trigger"]].copy()
        trig_slices_df["trigger_image"] = (
            trig_slices_df["tpc_trigger_offset"] / 10_000.0
        ).astype(int).clip(0, 49)
        trigger_image_map = dict(zip(trig_slices_df["slice_number"],
                                     trig_slices_df["trigger_image"]))

        # Write a temp parquet so streaming code can read it
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "images.parquet"
            images.to_parquet(img_path, index=False)
            agg = _stream_aggregate(img_path, triggered_slices, trigger_image_map)

    else:
        # ---- combined parquet directory: stream images ------------------
        slices_path = p / "slices.parquet"
        images_path = p / "images.parquet"
        if not slices_path.exists() or not images_path.exists():
            print(f"ERROR: expected slices.parquet and images.parquet in {p}")
            return

        print("Reading slices.parquet ...")
        slices = pq.read_table(
            slices_path,
            columns=["slice_number", "has_tpc_trigger", "tpc_trigger_offset"]
        ).to_pandas()

        n      = len(slices)
        n_trig = slices["has_tpc_trigger"].sum()
        print(f"Timeslices  : {n}")
        print(f"  triggered : {n_trig}  ({100*n_trig/n:.1f}%)")
        print(f"  baseline  : {n - n_trig}  ({100*(n-n_trig)/n:.1f}%)")
        print("(Image stats will be computed during streaming pass)")

        triggered_slices = set(slices.loc[slices["has_tpc_trigger"], "slice_number"])
        trig_df = slices[slices["has_tpc_trigger"]].copy()
        trig_df["trigger_image"] = (
            trig_df["tpc_trigger_offset"] / 10_000.0
        ).astype(int).clip(0, 49)
        trigger_image_map = dict(zip(trig_df["slice_number"],
                                     trig_df["trigger_image"]))

        print(f"Triggered slices : {len(triggered_slices)}")
        print(f"Streaming {images_path.name}  "
              f"({images_path.stat().st_size / 1e9:.1f} GB) "
              f"in batches of {args.batch_size:,} rows ...")

        agg = _stream_aggregate(images_path, triggered_slices, trigger_image_map)

    print("\nGenerating plots ...")
    plot_global_trigger_offset(slices, out_dir)
    plot_per_source_profiles(agg, out_dir)
    plot_trigger_aligned_profiles(agg, out_dir)
    plot_feature_distributions(agg, out_dir)

    print(f"\nAll plots saved to {out_dir}")





if __name__ == "__main__":
    main()