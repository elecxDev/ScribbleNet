"""
ScribbleNet - Dataset Module
PyTorch Dataset for CVL handwritten word images using TrOCR processor.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrOCRProcessor

from dip.augmentation import augment_pil_image
from dip.preprocessing import preprocess_pipeline
from utils.logger import get_logger

logger = get_logger("scribblenet.dataset")


class CVLWordDataset(Dataset):
    """
    PyTorch Dataset for CVL handwritten word images.

    Loads images and labels from a CSV file, applies optional preprocessing
    and augmentation, then encodes using the TrOCR processor.
    """

    def __init__(
        self,
        csv_path: str,
        processor: TrOCRProcessor,
        max_target_length: int = 32,
        apply_dip: bool = True,
        augment: bool = False,
        augment_config: Optional[Dict[str, Any]] = None,
        preprocess_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the CVL word dataset.

        Args:
            csv_path: Path to CSV file with columns [full_path, label, ...].
            processor: HuggingFace TrOCR processor.
            max_target_length: Maximum token length for labels.
            apply_dip: Whether to apply DIP preprocessing pipeline.
            augment: Whether to apply data augmentation.
            augment_config: Augmentation parameters.
            preprocess_config: DIP preprocessing parameters.
        """
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_target_length = max_target_length
        self.apply_dip = apply_dip
        self.augment = augment
        self.augment_config = augment_config or {}
        self.preprocess_config = preprocess_config or {}

        # Filter out rows with missing files
        valid_mask = self.df["full_path"].apply(lambda p: os.path.exists(str(p)))
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            logger.warning(
                "Skipping %d samples with missing image files.", invalid_count
            )
        self.df = self.df[valid_mask].reset_index(drop=True)

        # Filter out empty labels
        self.df = self.df[self.df["label"].notna() & (self.df["label"] != "")].reset_index(drop=True)

        logger.info(
            "Dataset loaded: %d samples from %s", len(self.df), csv_path
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'pixel_values' and 'labels' tensors.
        """
        row = self.df.iloc[idx]
        image_path = str(row["full_path"])
        label = str(row["label"])

        try:
            if self.apply_dip:
                image = preprocess_pipeline(
                    image_path,
                    grayscale=self.preprocess_config.get("grayscale", True),
                    noise_reduction=self.preprocess_config.get("noise_reduction", True),
                    adaptive_threshold=self.preprocess_config.get("adaptive_threshold", False),
                    contrast_enhancement=self.preprocess_config.get("contrast_enhancement", True),
                    gaussian_blur_kernel=self.preprocess_config.get("gaussian_blur_kernel", 3),
                )
            else:
                image = Image.open(image_path).convert("RGB")

            if self.augment:
                image = augment_pil_image(
                    image,
                    rotation_range=self.augment_config.get("rotation_range", 5),
                    scale_range=tuple(self.augment_config.get("scale_range", [0.9, 1.1])),
                    brightness_range=tuple(self.augment_config.get("brightness_range", [0.8, 1.2])),
                )

        except Exception as e:
            logger.error("Error loading image %s: %s", image_path, e)
            # Return a blank image as fallback
            image = Image.new("RGB", (384, 384), color=(255, 255, 255))

        # Process image
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Process label
        labels = self.processor.tokenizer(
            label,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so it's ignored by loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }


def create_dataloaders(
    config: Dict[str, Any],
    processor: TrOCRProcessor,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Full configuration dictionary.
        processor: HuggingFace TrOCR processor.

    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders.
    """
    from utils.file_manager import resolve_path

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    preprocess_cfg = config.get("preprocessing", {})
    augment_cfg = config.get("augmentation", {})

    loaders = {}

    for split, csv_key, batch_key, do_augment in [
        ("train", "train_csv", "batch_size", augment_cfg.get("enabled", False)),
        ("val", "val_csv", "batch_size", False),
        ("test", "test_csv", "batch_size", False),
    ]:
        csv_path = str(resolve_path(dataset_cfg[csv_key]))
        if not Path(csv_path).exists():
            logger.warning("CSV not found for %s split: %s", split, csv_path)
            continue

        ds = CVLWordDataset(
            csv_path=csv_path,
            processor=processor,
            max_target_length=config.get("model", {}).get("max_target_length", 32),
            apply_dip=True,
            augment=do_augment,
            augment_config=augment_cfg,
            preprocess_config=preprocess_cfg,
        )

        bs = training_cfg.get(batch_key, 16) if split == "train" else eval_cfg.get(batch_key, 32)
        num_workers = training_cfg.get("dataloader_num_workers", 4)

        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=bs,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

        logger.info(
            "Created %s DataLoader: %d samples, batch_size=%d",
            split, len(ds), bs,
        )

    return loaders
