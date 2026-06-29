import pyarrow.parquet as pq
import pandas as pd
import numpy as np

slices = pq.read_table("data/combined/amber-merged-002440-00040-00059.features.root/slices.parquet",
    columns=["slice_number","has_tpc_trigger","tpc_trigger_offset"]).to_pandas()

trig = slices[slices["has_tpc_trigger"]]
trigger_image = (trig["tpc_trigger_offset"] / 10_000.0).astype(int).clip(0, 49)
print("Trigger image index distribution:")
print(trigger_image.describe())
print(f"\nFraction of triggered slices where trigger falls in middle 20 images (15-35): "
      f"{((trigger_image >= 15) & (trigger_image <= 35)).mean():.1%}")
print(f"Fraction where trigger is in first or last 5 images: "
      f"{((trigger_image < 5) | (trigger_image > 44)).mean():.1%}")