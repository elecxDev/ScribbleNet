"""
ScribbleNet - Dataset Split Script
Writer-based 80/10/10 train/val/test split for the CVL dataset.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.file_manager import load_config, resolve_path
from utils.logger import get_logger, setup_logger

logger = get_logger("scribblenet.split")


def extract_csv_from_raw(config: Dict[str, Any]) -> str:
    """
    Extract word-level data from raw CVL dataset into a master CSV.

    Args:
        config: Configuration dictionary.

    Returns:
        Path to the generated master CSV.
    """
    dataset_cfg = config.get("dataset", {})
    raw_dir = str(resolve_path(dataset_cfg.get("raw_dataset_dir", "data/raw/cvl-database-1-1")))
    master_csv = str(resolve_path(dataset_cfg.get("master_csv", "data/processed/cvl_words_dataset.csv")))
    image_ext = dataset_cfg.get("image_extension", ".tif")

    rows = []
    for root, dirs, files in os.walk(raw_dir):
        # Only process 'words' directories
        if "words" not in root.lower():
            continue
        if any(skip in root.lower() for skip in ["lines", "pages", "xml"]):
            continue

        for file in files:
            if file.lower().endswith(image_ext):
                full_path = os.path.join(root, file)
                name_without_ext = os.path.splitext(file)[0]
                parts = name_without_ext.split("-")

                if len(parts) >= 5:
                    writer_id = parts[0]
                    text_id = parts[1]
                    line_id = parts[2]
                    word_index = parts[3]
                    label = "-".join(parts[4:])
                else:
                    continue

                rows.append({
                    "full_path": full_path,
                    "filename": file,
                    "writer_id": writer_id,
                    "text_id": text_id,
                    "line_id": line_id,
                    "word_index": word_index,
                    "label": label,
                })

    if not rows:
        logger.error("No image files found in %s", raw_dir)
        return ""

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    Path(master_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(master_csv, index=False)

    logger.info("Master CSV created: %d samples -> %s", len(df), master_csv)
    return master_csv


def split_dataset(config_path: Optional[str] = None) -> None:
    """
    Perform writer-based train/val/test split.

    Args:
        config_path: Path to configuration YAML.
    """
    config = load_config(config_path)
    dataset_cfg = config.get("dataset", {})

    master_csv = str(resolve_path(dataset_cfg.get("master_csv", "data/processed/cvl_words_dataset.csv")))

    # Check if master CSV exists, if not, try extracting
    if not Path(master_csv).exists():
        logger.info("Master CSV not found. Extracting from raw dataset...")
        master_csv = extract_csv_from_raw(config)
        if not master_csv:
            print("ERROR: Could not extract dataset. Ensure raw data is in data/raw/")
            return

    df = pd.read_csv(master_csv)
    logger.info("Loaded %d samples from master CSV.", len(df))

    # Get unique writers
    writers = df["writer_id"].unique()
    np.random.seed(42)  # Reproducibility
    np.random.shuffle(writers)

    # Split ratios
    train_ratio = dataset_cfg.get("train_ratio", 0.8)
    val_ratio = dataset_cfg.get("val_ratio", 0.1)

    train_split = int(train_ratio * len(writers))
    val_split = int((train_ratio + val_ratio) * len(writers))

    train_writers = writers[:train_split]
    val_writers = writers[train_split:val_split]
    test_writers = writers[val_split:]

    # Create splits
    train_df = df[df["writer_id"].isin(train_writers)]
    val_df = df[df["writer_id"].isin(val_writers)]
    test_df = df[df["writer_id"].isin(test_writers)]

    # Save splits
    splits_dir = resolve_path(config["paths"].get("splits_dir", "data/splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(splits_dir / "train.csv")
    val_path = str(splits_dir / "val.csv")
    test_path = str(splits_dir / "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 50)
    print("  Dataset Split Complete")
    print("=" * 50)
    print(f"  Total samples: {len(df)}")
    print(f"  Total writers: {len(writers)}")
    print(f"  Train: {len(train_df)} samples ({len(train_writers)} writers)")
    print(f"  Val:   {len(val_df)} samples ({len(val_writers)} writers)")
    print(f"  Test:  {len(test_df)} samples ({len(test_writers)} writers)")
    print(f"\n  Saved to: {splits_dir}")
    print("=" * 50)

    logger.info("Dataset split completed successfully.")


if __name__ == "__main__":
    split_dataset()
