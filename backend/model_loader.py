"""
ScribbleNet - Model Loader Module
Handles loading, configuring, and saving TrOCR models.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from utils.logger import get_logger

logger = get_logger("scribblenet.model_loader")


def get_device() -> torch.device:
    """
    Auto-detect the best available device (GPU or CPU).

    Returns:
        torch.device for training/inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no GPU detected).")
    return device


def load_model_and_processor(
    model_name: str = "microsoft/trocr-base-handwritten",
    freeze_encoder: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor, torch.device]:
    """
    Load the TrOCR model and processor.

    Args:
        model_name: HuggingFace model identifier.
        freeze_encoder: Whether to freeze the encoder layers.
        device: Target device. Auto-detected if None.

    Returns:
        Tuple of (model, processor, device).
    """
    if device is None:
        device = get_device()

    logger.info("Loading model: %s", model_name)
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configure decoder
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if freeze_encoder:
        logger.info("Freezing encoder layers.")
        for param in model.encoder.parameters():
            param.requires_grad = False
        # Log parameter counts
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Parameters: %d trainable / %d total (%.1f%%)",
            trainable, total, 100.0 * trainable / total,
        )

    model = model.to(device)
    logger.info("Model loaded and moved to %s.", device)

    return model, processor, device


def save_checkpoint(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    save_dir: str,
    epoch: int,
    optimizer: Optional[Any] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Save a model checkpoint.

    Args:
        model: Model to save.
        processor: Processor to save alongside.
        save_dir: Directory to save checkpoint in.
        epoch: Current epoch number.
        optimizer: Optional optimizer state to save.
        metrics: Optional metrics dictionary to include.

    Returns:
        Path to the saved checkpoint directory.
    """
    checkpoint_dir = Path(save_dir) / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(checkpoint_dir))
    processor.save_pretrained(str(checkpoint_dir))

    # Save additional metadata
    metadata = {"epoch": epoch, "metrics": metrics or {}}
    if optimizer is not None:
        torch.save(optimizer.state_dict(), str(checkpoint_dir / "optimizer.pt"))

    torch.save(metadata, str(checkpoint_dir / "metadata.pt"))

    logger.info("Checkpoint saved: %s", checkpoint_dir)
    return str(checkpoint_dir)


def save_best_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    save_dir: str,
) -> str:
    """
    Save the best model to the exported directory.

    Args:
        model: Best model to save.
        processor: Processor to save alongside.
        save_dir: Directory to save the exported model.

    Returns:
        Path to the saved model directory.
    """
    export_dir = Path(save_dir) / "best_model"
    export_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(export_dir))
    processor.save_pretrained(str(export_dir))

    logger.info("Best model exported to: %s", export_dir)
    return str(export_dir)


def load_checkpoint(
    checkpoint_dir: str,
    device: Optional[torch.device] = None,
) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor, torch.device]:
    """
    Load a model from a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory.
        device: Target device. Auto-detected if None.

    Returns:
        Tuple of (model, processor, device).
    """
    if device is None:
        device = get_device()

    logger.info("Loading checkpoint from: %s", checkpoint_dir)
    processor = TrOCRProcessor.from_pretrained(checkpoint_dir)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    model = model.to(device)

    logger.info("Checkpoint loaded successfully.")
    return model, processor, device
