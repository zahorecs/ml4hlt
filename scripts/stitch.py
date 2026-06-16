import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.loader import load_root
from src.config import DATA_DIR


def stitch(input_paths: list[Path], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    slices_out = output_path / "slices.parquet"
    images_out = output_path / "images.parquet"

    slices_writer = None
    images_writer = None

    total_slices    = 0
    total_triggered = 0
    n_files_ok      = 0

    for path in input_paths:
        print(f"  Loading {path.name} ...")
        try:
            slices, images = load_root(path)
        except Exception as e:
            print(f"  WARNING: skipping {path.name} — {e}")
            continue

        if len(slices) == 0:
            print(f"  WARNING: skipping {path.name} — empty file")
            continue

        source_file = path.stem
        slices["source_file"] = source_file
        images["source_file"] = source_file

        n      = len(slices)
        n_trig = slices["has_tpc_trigger"].sum()
        print(f"    {n} slices  {n_trig} triggered ({100*n_trig/n:.1f}%)")

        slices_table = pa.Table.from_pandas(slices, preserve_index=False)
        images_table = pa.Table.from_pandas(images, preserve_index=False)

        if slices_writer is None:
            slices_writer = pq.ParquetWriter(slices_out, slices_table.schema)
            images_writer = pq.ParquetWriter(images_out, images_table.schema)

        slices_writer.write_table(slices_table)
        images_writer.write_table(images_table)

        total_slices    += n
        total_triggered += n_trig
        n_files_ok      += 1

    if slices_writer is not None:
        slices_writer.close()
        images_writer.close()
    else:
        print("No files loaded. Exiting.")
        return

    print(f"\nCombined: {n_files_ok} files  {total_slices} slices  "
          f"{total_triggered} triggered ({100*total_triggered/total_slices:.1f}%)")
    print(f"\nWritten to {output_path}/")
    print(f"  slices.parquet : {slices_out.stat().st_size / 1e6:.1f} MB")
    print(f"  images.parquet : {images_out.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Stitch all .features.root files in a directory into parquet datasets.")
    parser.add_argument("input",
                        help="Directory containing .features.root files.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: data/combined/)")
    args = parser.parse_args()

    p = Path(args.input)
    if not p.is_dir():
        print(f"ERROR: {p} is not a directory.")
        return

    input_paths = sorted(p.glob("*.features.root"))
    if not input_paths:
        print(f"ERROR: no .features.root files found in {p}")
        return

    print(f"Found {len(input_paths)} file(s) in {p}:")
    for f in input_paths:
        print(f"  {f.name}")
    print()

    out_dir = Path(args.output) if args.output else DATA_DIR / "combined"
    stitch(input_paths, out_dir)


if __name__ == "__main__":
    main()