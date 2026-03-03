"""
ScribbleNet - Training Module
End-to-end training pipeline for the TrOCR model.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from backend.dataset import create_dataloaders
from backend.evaluate import evaluate_model
from backend.model_loader import (
    get_device,
    load_model_and_processor,
    save_best_model,
    save_checkpoint,
)
from utils.file_manager import ensure_directories, load_config, resolve_path
from utils.logger import get_logger, setup_logger

logger = get_logger("scribblenet.train")


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0,
    fp16: bool = True,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        dataloader: Training DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Target device.
        epoch: Current epoch number.
        max_grad_norm: Maximum gradient norm for clipping.
        fp16: Whether to use mixed precision training.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    scaler = torch.amp.GradScaler("cuda") if fp16 and device.type == "cuda" else None

    progress_bar = tqdm(
        dataloader, desc=f"Epoch {epoch + 1} [Train]", leave=False
    )

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Full training pipeline.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Dictionary with training results and metrics.
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    log_cfg = config.get("logging", {})
    setup_logger(
        name="scribblenet",
        log_file=str(resolve_path(log_cfg.get("file", "logs/scribblenet.log"))),
        level=log_cfg.get("level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("ScribbleNet Training Pipeline")
    logger.info("=" * 60)

    # Ensure directories
    ensure_directories(config)

    # Load model
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    model, processor, device = load_model_and_processor(
        model_name=model_cfg.get("name", "microsoft/trocr-base-handwritten"),
        freeze_encoder=model_cfg.get("freeze_encoder", True),
    )

    # Create data loaders
    loaders = create_dataloaders(config, processor)

    if "train" not in loaders:
        logger.error("Training data not found. Run dataset split first.")
        return {"error": "Training CSV not found."}

    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    # Optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_cfg.get("learning_rate", 5e-5),
        weight_decay=training_cfg.get("weight_decay", 0.01),
    )

    # Scheduler
    total_steps = len(train_loader) * training_cfg.get("num_epochs", 25)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_cfg.get("warmup_steps", 500),
        num_training_steps=total_steps,
    )

    # Training loop
    num_epochs = training_cfg.get("num_epochs", 25)
    patience = training_cfg.get("early_stopping_patience", 5)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    checkpoints_dir = str(resolve_path(config["paths"]["checkpoints_dir"]))
    exported_dir = str(resolve_path(config["paths"]["exported_dir"]))

    training_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
            fp16=training_cfg.get("fp16", True),
        )

        epoch_time = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f - Time: %.1fs",
            epoch + 1, num_epochs, train_loss, epoch_time,
        )

        # Validate
        val_loss = None
        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate_model(
                model=model,
                processor=processor,
                dataloader=val_loader,
                device=device,
            )
            val_loss = val_metrics.get("avg_loss", float("inf"))
            logger.info(
                "Epoch %d/%d - Val Loss: %.4f - Word Acc: %.4f",
                epoch + 1, num_epochs, val_loss,
                val_metrics.get("word_accuracy", 0.0),
            )

        # Record history
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # Early stopping check
        check_loss = val_loss if val_loss is not None else train_loss
        if check_loss < best_val_loss:
            best_val_loss = check_loss
            epochs_without_improvement = 0

            # Save best model
            save_best_model(model, processor, exported_dir)
            logger.info("New best model saved! Loss: %.4f", check_loss)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No improvement for %d epoch(s). Patience: %d/%d",
                epochs_without_improvement, epochs_without_improvement, patience,
            )

        # Save periodic checkpoint
        if training_cfg.get("save_best_only", True) is False or (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, processor, checkpoints_dir,
                epoch=epoch + 1, optimizer=optimizer,
                metrics=val_metrics,
            )

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    total_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds.", total_time)

    results = {
        "total_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
        "total_time_seconds": total_time,
        "history": training_history,
    }

    return results


if __name__ == "__main__":
    train()
