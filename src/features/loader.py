import uproot
import pandas as pd
from pathlib import Path


def load_root(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROOT file not found: {path}")

    with uproot.open(path) as f:
        slices = f["timeslices"].arrays(library="pd")
        images = f["images"].arrays(library="pd")

    slices = slices.astype({
        "slice_number":    "uint32",
        "spill_number":    "uint32",
        "slice_time":      "uint32",
        "run_number":      "uint32",
        "n_data_frames":   "uint32",
        "has_tpc_trigger": "bool",
    })

    images = images.astype({
        "slice_number":    "uint32",
        "source_id":       "uint32",
        "image_index":     "uint32",
        "hit_count":       "uint32",
        "active_channels": "uint32",
        "active_ports":    "uint32",
        "has_tpc_trigger": "bool",
    })

    return slices, images


def summary(slices: pd.DataFrame, images: pd.DataFrame) -> None:
    n          = len(slices)
    n_trig     = slices["has_tpc_trigger"].sum()
    n_sources  = images["source_id"].nunique()
    n_images   = len(images)

    print(f"Timeslices  : {n}")
    print(f"  triggered : {n_trig}  ({100*n_trig/n:.1f}%)")
    print(f"  baseline  : {n - n_trig}  ({100*(n-n_trig)/n:.1f}%)")
    print(f"Source IDs  : {n_sources}")
    print(f"Image rows  : {n_images}")
    print(f"Sources     : {sorted(images['source_id'].unique().tolist())}")

def load_combined(directory: str | Path,
                  slice_cols: list[str] | None = None,
                  image_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pyarrow.parquet as pq
    directory = Path(directory)

    slices = pq.read_table(directory / "slices.parquet",
                           columns=slice_cols).to_pandas()

    dataset = pq.ParquetFile(directory / "images.parquet")
    chunks = []
    for batch in dataset.iter_batches(batch_size=500_000, columns=image_cols):
        chunks.append(batch.to_pandas())
    images = pd.concat(chunks, ignore_index=True)

    return slices, images