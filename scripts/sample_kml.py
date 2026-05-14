"""
sample_kml.py
-------------
Creates a random sample of 1000 KML files from each of the 7 crop folders
found in culturas/culturas/culturas/ and copies them into a new
`culturas_sample/` directory, preserving the original folder structure.

Usage:
    python scripts/sample_kml.py
    python scripts/sample_kml.py --n 500          # custom sample size
    python scripts/sample_kml.py --seed 99        # reproducible shuffle
    python scripts/sample_kml.py --output my_out  # custom output dir
"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample KML files from crop folders.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("culturas/culturas/culturas"),
        help="Root directory containing the 7 crop sub-folders (default: culturas/culturas/culturas).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("culturas_sample"),
        help="Destination directory for sampled files (default: culturas_sample/).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of KML files to sample per folder (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def sample_folder(
    src_folder: Path,
    dst_folder: Path,
    n: int,
    rng: random.Random,
) -> int:
    """
    Randomly samples `n` KML files from `src_folder` and copies them to
    `dst_folder`. Returns the number of files actually copied.
    """
    kml_files = list(src_folder.glob("*.kml"))
    if not kml_files:
        print(f"  [WARN] No KML files found in: {src_folder}")
        return 0

    if len(kml_files) < n:
        print(
            f"  [WARN] {src_folder.name}: only {len(kml_files)} KML files available "
            f"(requested {n}). Copying all."
        )
        selected = kml_files
    else:
        selected = rng.sample(kml_files, n)

    dst_folder.mkdir(parents=True, exist_ok=True)
    for kml in selected:
        shutil.copy2(kml, dst_folder / kml.name)

    return len(selected)


def main() -> None:
    args = parse_args()

    # Resolve paths relative to project root (one level above scripts/)
    project_root = Path(__file__).resolve().parent.parent
    source = (project_root / args.source).resolve()
    output = (project_root / args.output).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    crop_folders = sorted([d for d in source.iterdir() if d.is_dir()])
    if not crop_folders:
        raise ValueError(f"No sub-folders found in: {source}")

    rng = random.Random(args.seed)

    print(f"Source : {source}")
    print(f"Output : {output}")
    print(f"Sample : {args.n} files per folder  |  seed={args.seed}")
    print(f"Folders: {len(crop_folders)}\n")

    total_copied = 0
    for folder in crop_folders:
        print(f"  → {folder.name}")
        dst = output / folder.name
        copied = sample_folder(folder, dst, args.n, rng)
        total_copied += copied
        print(f"     {copied} files copied to {dst}")

    print(f"\nDone! Total files copied: {total_copied}")


if __name__ == "__main__":
    main()
