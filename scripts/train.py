"""
train.py  —  Image-level Bayesian classifier (fully streaming)
==============================================================
Unit of classification: one (slice, image_index) pair — NOT per source.
Features are aggregated across all sources for each image slot, giving
one row per (slice × image) = 27k slices × 50 images = ~1.35M rows total.

Physics labelling:
  signal     : |image_index - trigger_image| <= drift_images  (default 10 = ±100µs)
  background : |image_index - trigger_image| >  drift_images
  Only images from triggered timeslices are used.

Features per (slice, image_index) — summed/averaged across all sources:
  total_hits, mean_hits_per_source, max_hits_per_source,
  active_channels_sum, active_ports_sum,
  mean_port_entropy, mean_channel_spread, mean_hit_time_range,
  n_active_sources  (how many sources fired at all)
  rel_image         (signed distance from trigger image — physics timing)

Usage
-----
  python train.py data/combined/
  python train.py data/combined/ --drift-images 10 --dim-reduction lda
"""

import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from scipy import linalg
import time
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.evaluation import (
    plot_roc, plot_score_distribution,
    plot_calibration, print_confusion,
)
from src.config import PLOTS_DIR

BATCH_SIZE   = 500_000
DRIFT_IMAGES = 10        # ±10 images = ±100 µs = TPC drift time

# Per-image aggregated features (summed/averaged across sources)
FEATURE_COLS = [
    "rel_image",           # physics timing feature
    "total_hits",          # sum of hit_count across sources
    "mean_hits",           # mean hit_count across sources
    "max_hits",            # max hit_count across sources
    "n_active_sources",    # how many sources had any hits
    "active_channels_sum", # sum of active_channels across sources
    "active_ports_sum",    # sum of active_ports across sources
    "mean_port_entropy",   # mean port_entropy across sources
    "mean_channel_spread", # mean channel_spread across sources
    "mean_hit_time_range", # mean hit_time_range across sources
]
N_FEAT = len(FEATURE_COLS)

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Streaming sufficient-statistics Gaussian classifier
# ---------------------------------------------------------------------------

class StreamingGaussianClassifier:
    def __init__(self, prior_strength: float = 1.0, shared_covariance: bool = True):
        self.prior_strength    = prior_strength
        self.shared_covariance = shared_covariance
        self.mu_ = {}; self.sigma_ = {}; self.log_prior_ = {}
        self.classes_ = None

    def new_accumulators(self, d: int) -> dict:
        return {k: {"n": 0, "sum": np.zeros(d), "M2": np.zeros((d, d))}
                for k in (0, 1)}

    def update(self, acc: dict, X: np.ndarray, y: np.ndarray) -> None:
        for k in (0, 1):
            mask = y == k
            if not mask.any():
                continue
            Xk = X[mask]
            acc[k]["n"]   += len(Xk)
            acc[k]["sum"] += Xk.sum(axis=0)
            acc[k]["M2"]  += Xk.T @ Xk

    def finalise(self, acc: dict) -> None:
        d       = acc[0]["sum"].shape[0]
        n_total = acc[0]["n"] + acc[1]["n"]
        self.classes_ = np.array([0, 1])
        kappa_0 = self.prior_strength

        for k in (0, 1):
            nk      = acc[k]["n"]
            kappa_n = kappa_0 + nk
            mu_n    = acc[k]["sum"] / kappa_n
            self.mu_[k]       = mu_n
            self.log_prior_[k] = np.log(nk / n_total) if nk > 0 else -np.inf

        nu_0 = d + 2
        if self.shared_covariance:
            S = np.zeros((d, d))
            for k in (0, 1):
                S += acc[k]["M2"] - acc[k]["n"] * np.outer(self.mu_[k], self.mu_[k])
            sigma = (np.eye(d) + S) / (n_total + nu_0 - d - 1)
            for k in (0, 1):
                self.sigma_[k] = sigma
        else:
            for k in (0, 1):
                nk  = acc[k]["n"]
                S   = acc[k]["M2"] - nk * np.outer(self.mu_[k], self.mu_[k])
                self.sigma_[k] = (np.eye(d) + S) / (nk + nu_0 - d - 1)

    def predict_proba_signal(self, X: np.ndarray) -> np.ndarray:
        log_p = np.column_stack([
            self._log_lik(X, k) + self.log_prior_[k] for k in (0, 1)
        ])
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p /= p.sum(axis=1, keepdims=True)
        return p[:, 1]

    def _log_lik(self, X: np.ndarray, k: int) -> np.ndarray:
        mu = self.mu_[k]; sigma = self.sigma_[k]; d = X.shape[1]
        try:
            L = linalg.cholesky(sigma, lower=True)
        except linalg.LinAlgError:
            sigma = sigma + 1e-6 * np.eye(d)
            L = linalg.cholesky(sigma, lower=True)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        alpha   = linalg.solve_triangular(L, (X - mu).T, lower=True)
        maha    = np.sum(alpha**2, axis=0)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


class StreamingStandardScaler:
    def fit_from_stats(self, n, sum_x, sum_x2):
        self.mean_ = sum_x / n
        self.std_  = np.sqrt(np.maximum(sum_x2 / n - self.mean_**2, 0))
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


# ---------------------------------------------------------------------------
# Core: build (slice, image_index) aggregated features from one batch
# ---------------------------------------------------------------------------

def _aggregate_batch(batch, in_fold_mask, trig_img_arr, max_sn, drift_images):
    """
    Aggregate one parquet batch across sources to produce
    one row per (slice_number, image_index).

    Returns dict of numpy arrays, all length = n_unique_(sn, img) pairs.
    """
    sn_np  = batch.column("slice_number").to_numpy(zero_copy_only=False).astype(np.int32)[in_fold_mask]
    img_np = batch.column("image_index").to_numpy(zero_copy_only=False).astype(np.int32)[in_fold_mask]
    hc_np  = batch.column("hit_count").to_numpy(zero_copy_only=False).astype(np.float32)[in_fold_mask]
    ac_np  = batch.column("active_channels").to_numpy(zero_copy_only=False).astype(np.float32)[in_fold_mask]
    ap_np  = batch.column("active_ports").to_numpy(zero_copy_only=False).astype(np.float32)[in_fold_mask]
    pe_np  = batch.column("port_entropy").to_numpy(zero_copy_only=False).astype(np.float32)[in_fold_mask]
    cs_np  = batch.column("channel_spread").to_numpy(zero_copy_only=False).astype(np.float32)[in_fold_mask]
    htr_np = batch.column("hit_time_range").to_numpy(zero_copy_only=False).astype(np.float32)[in_fold_mask]

    # Encode (sn, img) pairs into a single key for bincount aggregation
    unique_sn,  sn_inv  = np.unique(sn_np,  return_inverse=True)
    unique_img, img_inv = np.unique(img_np, return_inverse=True)

    n_sn  = len(unique_sn)
    n_img = len(unique_img)

    # Use a 2D key: sn_rank * n_img + img_rank
    keys   = sn_inv * n_img + img_inv
    n_bins = n_sn * n_img

    total_hits = np.bincount(keys, weights=hc_np,  minlength=n_bins).reshape(n_sn, n_img)
    sum_ac     = np.bincount(keys, weights=ac_np,  minlength=n_bins).reshape(n_sn, n_img)
    sum_ap     = np.bincount(keys, weights=ap_np,  minlength=n_bins).reshape(n_sn, n_img)
    sum_pe     = np.bincount(keys, weights=pe_np,  minlength=n_bins).reshape(n_sn, n_img)
    sum_cs     = np.bincount(keys, weights=cs_np,  minlength=n_bins).reshape(n_sn, n_img)
    sum_htr    = np.bincount(keys, weights=htr_np, minlength=n_bins).reshape(n_sn, n_img)
    cnt        = np.bincount(keys,                 minlength=n_bins).reshape(n_sn, n_img).astype(np.float32)

    # Max hits: need groupby — use pandas on the small unique-key space
    max_hits = pd.DataFrame({"k": keys, "hc": hc_np}).groupby("k")["hc"].max()
    max_hits_2d = np.zeros(n_bins, dtype=np.float32)
    max_hits_2d[max_hits.index.values] = max_hits.values
    max_hits_2d = max_hits_2d.reshape(n_sn, n_img)

    safe_cnt = np.where(cnt > 0, cnt, 1.0)

    # Flatten to rows
    sn_out  = np.repeat(unique_sn,  n_img)
    img_out = np.tile  (unique_img, n_sn)
    valid   = cnt.ravel() > 0   # only (sn, img) pairs that actually appear

    sn_out  = sn_out [valid]
    img_out = img_out[valid]

    trig_img = trig_img_arr[np.clip(sn_out, 0, max_sn-1)]
    rel      = (img_out - trig_img).astype(np.float32)
    label    = (np.abs(img_out - trig_img) <= drift_images).astype(np.int8)

    n_rows = valid.sum()
    X_batch = np.empty((n_rows, N_FEAT), dtype=np.float32)
    c = cnt.ravel()[valid]
    sc = safe_cnt.ravel()[valid]

    X_batch[:, 0]  = rel
    X_batch[:, 1]  = total_hits.ravel()[valid]
    X_batch[:, 2]  = total_hits.ravel()[valid] / sc
    X_batch[:, 3]  = max_hits_2d.ravel()[valid]
    X_batch[:, 4]  = c                                    # n_active_sources
    X_batch[:, 5]  = sum_ac.ravel() [valid]
    X_batch[:, 6]  = sum_ap.ravel() [valid]
    X_batch[:, 7]  = sum_pe.ravel() [valid] / sc
    X_batch[:, 8]  = sum_cs.ravel() [valid] / sc
    X_batch[:, 9]  = sum_htr.ravel()[valid] / sc

    return X_batch, label, sn_out


# ---------------------------------------------------------------------------
# One streaming pass over images.parquet
# ---------------------------------------------------------------------------

NEEDED_COLS = ["slice_number", "image_index", "hit_count",
               "active_channels", "active_ports", "port_entropy",
               "channel_spread", "hit_time_range"]

def stream_pass(images_path, trig_img_arr, trig_flag, max_sn,
                drift_images, slice_set, scaler, lda_w, mode, clf=None):
    """
    mode: 'fit_scaler' | 'fit_clf' | 'predict'
    Returns:
      fit_scaler -> (n, sum_x, sum_x2)
      fit_clf    -> acc dict
      predict    -> (y_true, y_prob)
    """
    dataset    = pq.ParquetFile(images_path)
    total_rows = dataset.metadata.num_rows
    rows_done  = 0
    t0         = time.time()

    if mode == "fit_scaler":
        n_acc = 0; sum_x = np.zeros(N_FEAT); sum_x2 = np.zeros(N_FEAT)
    elif mode == "fit_clf":
        d   = 1 if lda_w is not None else N_FEAT
        acc = clf.new_accumulators(d)
    else:
        y_true_l = []; y_prob_l = []

    for batch in dataset.iter_batches(batch_size=BATCH_SIZE, columns=NEEDED_COLS):
        rows_done += batch.num_rows
        elapsed    = time.time() - t0
        rate       = rows_done / elapsed if elapsed > 0 else 1
        eta        = (total_rows - rows_done) / rate / 60
        print(f"    [{_ts()}] {rows_done:>12,}/{total_rows:,} "
              f"({100*rows_done/total_rows:4.1f}%) "
              f"{rate/1e6:.1f}M/s ETA {eta:.1f}m", flush=True)

        sn_np = batch.column("slice_number").to_numpy(zero_copy_only=False).astype(np.int32)
        safe  = np.clip(sn_np, 0, max_sn - 1)
        mask  = trig_flag[safe] & np.isin(sn_np, list(slice_set))
        if not mask.any():
            continue

        # Aggregate across sources → one row per (slice, image)
        X_raw, label, sn_out = _aggregate_batch(
            batch, mask, trig_img_arr, max_sn, drift_images)

        if mode == "fit_scaler":
            n_acc  += len(X_raw)
            Xf      = X_raw.astype(np.float64)
            sum_x  += Xf.sum(axis=0)
            sum_x2 += (Xf**2).sum(axis=0)
            continue

        X_sc = scaler.transform(X_raw.astype(np.float64))
        if lda_w is not None:
            X_sc = (X_sc @ lda_w)[:, None]

        if mode == "fit_clf":
            clf.update(acc, X_sc, label)
        else:
            probs = clf.predict_proba_signal(X_sc)
            y_true_l.append(label)
            y_prob_l.append(probs.astype(np.float32))

    if mode == "fit_scaler":
        return n_acc, sum_x, sum_x2
    elif mode == "fit_clf":
        return acc
    else:
        return np.concatenate(y_true_l), np.concatenate(y_prob_l)


# ---------------------------------------------------------------------------
# CV fold builder
# ---------------------------------------------------------------------------

def build_fold_splits(slices_path, n_splits=5, seed=42):
    slices    = pq.read_table(slices_path,
                    columns=["slice_number","has_tpc_trigger"]).to_pandas()
    trig_sn   = slices[slices["has_tpc_trigger"]]["slice_number"].values.copy()
    rng       = np.random.default_rng(seed)
    rng.shuffle(trig_sn)
    folds     = [set() for _ in range(n_splits)]
    for i, sn in enumerate(trig_sn):
        folds[i % n_splits].add(int(sn))
    splits = []
    for v in range(n_splits):
        val   = folds[v]
        train = set().union(*[folds[i] for i in range(n_splits) if i != v])
        splits.append((train, val))
    return splits



# ---------------------------------------------------------------------------
# Per-source diagnostic
# ---------------------------------------------------------------------------

PER_SOURCE_FEATURES = [
    "hit_count", "active_channels", "active_ports",
    "port_entropy", "channel_spread", "hit_time_range",
    "mean_hit_time", "var_hit_time",
]
N_SRC_FEAT = len(PER_SOURCE_FEATURES) + 1  # +1 for rel_image


def stream_per_source(images_path, trig_img_arr, trig_flag, max_sn,
                      drift_images, slice_set, mode,
                      scalers=None, clfs=None):
    """
    Stream images keeping each source separate.
    mode: 'fit_scaler' | 'fit_clf' | 'predict'

    Returns:
      fit_scaler -> dict: sid -> (n, sum_x, sum_x2)
      fit_clf    -> dict: sid -> acc
      predict    -> dict: sid -> (y_true, y_prob)
    """
    needed = ["slice_number", "source_id", "image_index"] + PER_SOURCE_FEATURES
    dataset = pq.ParquetFile(images_path)
    total_rows = dataset.metadata.num_rows
    rows_done = 0
    t0 = time.time()

    if mode == "fit_scaler":
        result = {}   # sid -> [n, sum_x, sum_x2]
    elif mode == "fit_clf":
        result = {}   # sid -> acc (initialised on first encounter)
    else:
        y_true_d = {}; y_prob_d = {}

    for batch in dataset.iter_batches(batch_size=BATCH_SIZE, columns=needed):
        rows_done += batch.num_rows
        elapsed = time.time() - t0
        rate    = rows_done / elapsed if elapsed > 0 else 1
        eta     = (total_rows - rows_done) / rate / 60
        print(f"    [{_ts()}] {rows_done:>12,}/{total_rows:,} "
              f"({100*rows_done/total_rows:4.1f}%) "
              f"{rate/1e6:.1f}M/s ETA {eta:.1f}m", flush=True)

        sn_np  = batch.column("slice_number").to_numpy(zero_copy_only=False).astype(np.int32)
        safe   = np.clip(sn_np, 0, max_sn - 1)
        mask   = trig_flag[safe] & np.isin(sn_np, list(slice_set))
        if not mask.any():
            continue

        sn_f   = sn_np[mask]
        sid_f  = batch.column("source_id").to_numpy(zero_copy_only=False).astype(np.int32)[mask]
        img_f  = batch.column("image_index").to_numpy(zero_copy_only=False).astype(np.int32)[mask]

        trig_img_f = trig_img_arr[np.clip(sn_f, 0, max_sn-1)]
        rel        = (img_f - trig_img_f).astype(np.float32)
        label      = (np.abs(img_f - trig_img_f) <= drift_images).astype(np.int8)

        # Build feature array: rel_image + per-source features
        feat_parts = [rel[:, None]]
        for feat in PER_SOURCE_FEATURES:
            col = batch.column(feat)
            feat_parts.append(col.to_numpy(zero_copy_only=False).astype(np.float32)[mask, None])
        X_raw = np.hstack(feat_parts)  # (n, N_SRC_FEAT)

        # Process each source separately
        for sid in np.unique(sid_f):
            sm = sid_f == sid
            Xs = X_raw[sm]
            ys = label[sm]
            sid = int(sid)

            if mode == "fit_scaler":
                if sid not in result:
                    result[sid] = [0, np.zeros(N_SRC_FEAT), np.zeros(N_SRC_FEAT)]
                result[sid][0] += len(Xs)
                Xd = Xs.astype(np.float64)
                result[sid][1] += Xd.sum(axis=0)
                result[sid][2] += (Xd**2).sum(axis=0)

            elif mode == "fit_clf":
                if sid not in result:
                    result[sid] = clfs[sid].new_accumulators(N_SRC_FEAT)
                sc = scalers[sid].transform(Xs.astype(np.float64))
                clfs[sid].update(result[sid], sc, ys)

            else:  # predict
                sc    = scalers[sid].transform(Xs.astype(np.float64))
                probs = clfs[sid].predict_proba_signal(sc)
                if sid not in y_true_d:
                    y_true_d[sid] = []; y_prob_d[sid] = []
                y_true_d[sid].append(ys)
                y_prob_d[sid].append(probs.astype(np.float32))

    if mode == "fit_scaler":
        return result
    elif mode == "fit_clf":
        return result
    else:
        return {sid: (np.concatenate(y_true_d[sid]),
                      np.concatenate(y_prob_d[sid]))
                for sid in y_true_d}


def run_per_source_diagnostic(images_path, trig_img_arr, trig_flag, max_sn,
                               drift_images, all_trig, prior_strength, shared_cov):
    """
    For each source_id independently:
      1. Fit scaler on all triggered slices
      2. Fit Gaussian classifier on 80% of slices
      3. Evaluate AUC on remaining 20%
      4. Report mean separation and AUC

    Uses a single train/val split (not full CV) since this is diagnostic only.
    """
    print(f"\n[{_ts()}] === PER-SOURCE DIAGNOSTIC ===")
    print(f"  Training one classifier per source on raw per-image features")
    print(f"  (no source aggregation — each source evaluated independently)")

    # Simple 80/20 split
    all_sn  = sorted(all_trig)
    rng     = np.random.default_rng(99)
    rng.shuffle(all_sn := np.array(all_sn))
    cut     = int(0.8 * len(all_sn))
    train_s = set(all_sn[:cut].tolist())
    val_s   = set(all_sn[cut:].tolist())
    print(f"  Train: {len(train_s)} slices  Val: {len(val_s)} slices")

    # Pass 1: fit scalers
    print(f"\n[{_ts()}]  Pass 1/3: fitting per-source scalers ...")
    scaler_stats = stream_per_source(images_path, trig_img_arr, trig_flag, max_sn,
                                     drift_images, all_trig, "fit_scaler")
    scalers = {}
    for sid, (n, sx, sx2) in scaler_stats.items():
        scalers[sid] = StreamingStandardScaler().fit_from_stats(n, sx, sx2)
    source_ids = sorted(scalers.keys())
    print(f"  Found {len(source_ids)} sources: {source_ids}")

    # Initialise classifiers
    clfs = {sid: StreamingGaussianClassifier(prior_strength, shared_cov)
            for sid in source_ids}

    # Pass 2: fit classifiers
    print(f"\n[{_ts()}]  Pass 2/3: fitting per-source classifiers (training fold) ...")
    acc_dict = stream_per_source(images_path, trig_img_arr, trig_flag, max_sn,
                                 drift_images, train_s, "fit_clf",
                                 scalers=scalers, clfs=clfs)
    for sid in source_ids:
        if sid in acc_dict:
            clfs[sid].finalise(acc_dict[sid])

    # Pass 3: predict
    print(f"\n[{_ts()}]  Pass 3/3: predicting on validation fold ...")
    preds = stream_per_source(images_path, trig_img_arr, trig_flag, max_sn,
                              drift_images, val_s, "predict",
                              scalers=scalers, clfs=clfs)

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Source':>8}  {'N_val':>9}  {'Sig%':>6}  "
          f"{'||mu||':>7}  {'AUC':>7}  {'Verdict'}")
    print(f"{'='*65}")

    results = []
    for sid in source_ids:
        if sid not in preds:
            continue
        y_true, y_prob = preds[sid]
        auc  = _quick_auc(y_true, y_prob)
        mu_d = np.linalg.norm(clfs[sid].mu_[1] - clfs[sid].mu_[0]) if clfs[sid].classes_ is not None else 0.0
        sig_frac = 100 * y_true.mean()
        verdict = ("*** SIGNAL ***" if auc > 0.65
                   else "some signal"  if auc > 0.58
                   else "weak"         if auc > 0.54
                   else "noise")
        print(f"  {sid:>6}  {len(y_true):>9,}  {sig_frac:>5.1f}%  "
              f"{mu_d:>7.4f}  {auc:>7.4f}  {verdict}")
        results.append((sid, auc, mu_d))

    print(f"{'='*65}")
    best = max(results, key=lambda x: x[1])
    worst = min(results, key=lambda x: x[1])
    print(f"  Best source : {best[0]}  (AUC={best[1]:.4f}  ||mu||={best[2]:.4f})")
    print(f"  Worst source: {worst[0]}  (AUC={worst[1]:.4f}  ||mu||={worst[2]:.4f})")
    print(f"  Mean AUC    : {np.mean([r[1] for r in results]):.4f}")
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_path, plots_dir, drift_images, dim_reduction,
        prior_strength, shared_cov, pca_threshold):

    t_start = time.time()
    slices_path = data_path / "slices.parquet"
    images_path = data_path / "images.parquet"

    print(f"[{_ts()}] Reading slices.parquet ...")
    slices = pq.read_table(slices_path,
        columns=["slice_number","has_tpc_trigger","tpc_trigger_offset"]).to_pandas()

    n_total = len(slices); n_trig = slices["has_tpc_trigger"].sum()
    print(f"  Timeslices: {n_total}  triggered: {n_trig} ({100*n_trig/n_total:.1f}%)")

    trig_df = slices[slices["has_tpc_trigger"]].copy()
    trig_df["trigger_image"] = (trig_df["tpc_trigger_offset"]/10_000.0).astype(int).clip(0,49)

    max_sn       = int(slices["slice_number"].max()) + 1
    trig_img_arr = np.full(max_sn, -1, dtype=np.int32)
    trig_flag    = np.zeros(max_sn, dtype=bool)
    for _, row in trig_df[["slice_number","trigger_image"]].iterrows():
        sn = int(row["slice_number"])
        trig_img_arr[sn] = int(row["trigger_image"])
        trig_flag[sn]    = True

    all_trig = set(trig_df["slice_number"].values.tolist())
    print(f"  Signal window: ±{drift_images} images = ±{drift_images*10} µs")
    print(f"  Expected rows: ~{len(all_trig)*50/1e6:.2f}M  (vs 204M before source aggregation)")

    # --- Fit scaler on all triggered images ------------------------------
    print(f"\n[{_ts()}] Fitting StandardScaler (all triggered slices) ...")
    t0 = time.time()
    n_sc, sum_x, sum_x2 = stream_pass(
        images_path, trig_img_arr, trig_flag, max_sn,
        drift_images, all_trig, None, None, "fit_scaler")
    scaler = StreamingStandardScaler().fit_from_stats(n_sc, sum_x, sum_x2)
    print(f"  Scaler fitted on {n_sc:,} (slice,image) pairs  [{time.time()-t0:.0f}s]")

    # --- CV --------------------------------------------------------------
    print(f"\n[{_ts()}] Building 5-fold CV splits ...")
    fold_splits = build_fold_splits(slices_path)

    all_true = []; all_prob = []

    for fold, (train_set, val_set) in enumerate(fold_splits):
        t_fold = time.time()
        print(f"\n{'='*60}")
        print(f"[{_ts()}] FOLD {fold+1}/5  "
              f"train={len(train_set)} slices  val={len(val_set)} slices")
        print(f"{'='*60}")

        lda_w = None

        if dim_reduction == "lda":
            print(f"[{_ts()}]   LDA direction pass ...")
            t0    = time.time()
            tmp   = StreamingGaussianClassifier(prior_strength, shared_cov)
            a_lda = stream_pass(images_path, trig_img_arr, trig_flag, max_sn,
                                drift_images, train_set, scaler, None, "fit_clf", tmp)
            tmp.finalise(a_lda)
            n_tr  = a_lda[0]["n"] + a_lda[1]["n"]
            S_w   = np.zeros((N_FEAT, N_FEAT))
            for k in (0,1):
                nk = a_lda[k]["n"]
                S_w += a_lda[k]["M2"] - nk * np.outer(tmp.mu_[k], tmp.mu_[k])
            S_w /= (n_tr - 2); S_w += 1e-6 * np.eye(N_FEAT)
            w = linalg.solve(S_w, tmp.mu_[1]-tmp.mu_[0], assume_a="pos")
            lda_w = w / np.linalg.norm(w)
            print(f"  LDA done [{time.time()-t0:.0f}s]")

        print(f"[{_ts()}]   Fit pass (training fold) ...")
        t0  = time.time()
        clf = StreamingGaussianClassifier(prior_strength, shared_cov)
        acc = stream_pass(images_path, trig_img_arr, trig_flag, max_sn,
                          drift_images, train_set, scaler, lda_w, "fit_clf", clf)
        clf.finalise(acc)
        n0, n1 = acc[0]["n"], acc[1]["n"]
        print(f"  Fitted: bg={n0:,}  sig={n1:,}  frac={100*n1/(n0+n1):.1f}%  "
              f"[{time.time()-t0:.0f}s]")

        print(f"[{_ts()}]   ||mu_sig - mu_bg|| = "
              f"{np.linalg.norm(clf.mu_[1]-clf.mu_[0]):.4f}")

        print(f"[{_ts()}]   Predict pass (validation fold) ...")
        t0 = time.time()
        y_true, y_prob = stream_pass(images_path, trig_img_arr, trig_flag, max_sn,
                                     drift_images, val_set, scaler, lda_w,
                                     "predict", clf)
        fold_auc = _quick_auc(y_true, y_prob)
        n_sig    = int(y_true.sum())
        print(f"  Predict done [{time.time()-t0:.0f}s]")
        print(f"  Fold {fold+1}: AUC={fold_auc:.4f}  "
              f"n_val={len(y_true):,}  sig={n_sig:,} [{100*n_sig/len(y_true):.1f}%]")
        print(f"  Fold {fold+1} total time: {(time.time()-t_fold)/60:.1f} min")

        all_true.append(y_true)
        all_prob.append(y_prob)

    y_true_all = np.concatenate(all_true)
    y_prob_all = np.concatenate(all_prob).astype(np.float64)
    print(f"\n[{_ts()}] Total training time: {(time.time()-t_start)/60:.1f} min")
    print(f"Combined val set: {len(y_true_all):,} rows  "
          f"signal={y_true_all.sum():,} [{100*y_true_all.mean():.1f}%]")

    # --- Plots -----------------------------------------------------------
    print(f"\n[{_ts()}] Generating evaluation plots ...")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(y_true_all, y_prob_all, plots_dir)
    plot_score_distribution(y_true_all, y_prob_all, plots_dir)
    plot_calibration(y_true_all, y_prob_all, plots_dir)

    for thresh in [0.3, 0.5, 0.7]:
        y_pred = (y_prob_all >= thresh).astype(int)
        print_confusion(y_true_all, y_pred, threshold=thresh)

    print(f"[{_ts()}] All plots saved to {plots_dir}")

    # --- Per-source diagnostic -------------------------------------------
    run_per_source_diagnostic(
        images_path, trig_img_arr, trig_flag, max_sn,
        drift_images, all_trig, prior_strength, shared_cov
    )


def _quick_auc(y_true, y_score):
    order  = np.argsort(y_score)[::-1]
    y_sort = y_true[order]
    n_pos  = int(y_true.sum()); n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tp = 0; auc = 0.0
    for lbl in y_sort:
        if lbl == 1: tp += 1
        else:        auc += tp
    return auc / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--plots-dir",      default=None)
    parser.add_argument("--drift-images",   type=int,   default=DRIFT_IMAGES)
    parser.add_argument("--dim-reduction",  default="none", choices=["none","lda"])
    parser.add_argument("--pca-threshold",  type=float, default=0.95)
    parser.add_argument("--prior-strength", type=float, default=1.0)
    parser.add_argument("--no-shared-cov",  action="store_true")
    args = parser.parse_args()

    run(data_path      = Path(args.input),
        plots_dir      = Path(args.plots_dir) if args.plots_dir else PLOTS_DIR,
        drift_images   = args.drift_images,
        dim_reduction  = args.dim_reduction,
        pca_threshold  = args.pca_threshold,
        prior_strength = args.prior_strength,
        shared_cov     = not args.no_shared_cov)

if __name__ == "__main__":
    main()