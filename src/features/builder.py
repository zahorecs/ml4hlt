import numpy as np
import pandas as pd


def build_feature_matrix(
    slices: pd.DataFrame,
    images: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot the images table into one row per timeslice.

    For each (timeslice, source_id) we compute:
      - total_hits
      - mean_hits_per_image
      - std_hits_per_image
      - max_hits_per_image
      - mean_active_channels
      - mean_port_entropy

    These are then flattened into columns named
    <feature>_<source_id>, giving one wide row per timeslice.

    The result is joined with the slice-level label
    (has_tpc_trigger, tpc_trigger_offset).
    """
    agg = (
        images
        .groupby(["slice_number", "source_id"])
        .agg(
            total_hits        = ("hit_count",       "sum"),
            mean_hits         = ("hit_count",       "mean"),
            std_hits          = ("hit_count",       "std"),
            max_hits          = ("hit_count",       "max"),
            mean_active_ch    = ("active_channels", "mean"),
            mean_port_entropy = ("port_entropy",    "mean"),
        )
        .reset_index()
    )

    agg["std_hits"] = agg["std_hits"].fillna(0.0)

    wide = agg.pivot(index="slice_number", columns="source_id")
    wide.columns = [f"{feat}_{sid}" for feat, sid in wide.columns]
    wide = wide.reset_index()

    labels = slices[["slice_number", "has_tpc_trigger", "tpc_trigger_offset"]].copy()
    result = labels.merge(wide, on="slice_number", how="inner")

    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"slice_number", "has_tpc_trigger", "tpc_trigger_offset"}
    return [c for c in df.columns if c not in exclude]


def split_features_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values.astype(np.float64)
    y = df["has_tpc_trigger"].values.astype(int)
    return X, y