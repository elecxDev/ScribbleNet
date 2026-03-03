"""
ScribbleNet - Evaluation Module
Model evaluation with comprehensive metrics.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.logger import get_logger
from utils.metrics import compute_all_metrics

logger = get_logger("scribblenet.evaluate")


def evaluate_model(
    model: VisionEncoderDecoderModel,
    processor: TrOCRProcessor,
    dataloader: DataLoader,
    device: torch.device,
    num_beams: int = 4,
    max_target_length: int = 32,
) -> Dict[str, Any]:
    """
    Evaluate the model on a given dataset.

    Args:
        model: The trained model.
        processor: TrOCR processor for decoding.
        dataloader: Evaluation DataLoader.
        device: Target device.
        num_beams: Number of beams for beam search decoding.
        max_target_length: Maximum generated sequence length.

    Returns:
        Dictionary containing loss and all metrics.
    """
    model.eval()
    all_predictions: List[str] = []
    all_targets: List[str] = []
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Compute loss
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1

            # Generate predictions
            generated_ids = model.generate(
                pixel_values,
                max_length=max_target_length,
                num_beams=num_beams,
            )

            # Decode predictions
            pred_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            all_predictions.extend(pred_texts)

            # Decode targets (replace -100 with pad token for decoding)
            label_ids = labels.clone()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            target_texts = processor.tokenizer.batch_decode(
                label_ids, skip_special_tokens=True
            )
            all_targets.extend(target_texts)

    avg_loss = total_loss / max(num_batches, 1)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets)
    metrics["avg_loss"] = avg_loss
    metrics["num_samples"] = len(all_predictions)

    logger.info("Evaluation Results:")
    logger.info("  Samples: %d", metrics["num_samples"])
    logger.info("  Avg Loss: %.4f", avg_loss)
    logger.info("  Word Accuracy: %.4f", metrics["word_accuracy"])
    logger.info("  Char Accuracy: %.4f", metrics["character_accuracy"])
    logger.info("  Avg Levenshtein: %.4f", metrics["avg_levenshtein_distance"])
    logger.info("  Avg NED: %.4f", metrics["avg_normalized_edit_distance"])

    return metrics


def evaluate_from_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run evaluation using configuration file.

    Args:
        config_path: Path to configuration YAML.

    Returns:
        Dictionary with evaluation metrics.
    """
    from backend.dataset import create_dataloaders
    from backend.model_loader import load_checkpoint
    from utils.file_manager import load_config, resolve_path

    config = load_config(config_path)
    eval_cfg = config.get("evaluation", {})
    model_cfg = config.get("model", {})

    # Try to load best model, fall back to base model
    exported_dir = resolve_path(config["paths"]["exported_dir"])
    best_model_dir = exported_dir / "best_model"

    if best_model_dir.exists():
        logger.info("Loading best model from: %s", best_model_dir)
        model, processor, device = load_checkpoint(str(best_model_dir))
    else:
        logger.info("No fine-tuned model found. Loading base model.")
        from backend.model_loader import load_model_and_processor
        model, processor, device = load_model_and_processor(
            model_name=model_cfg.get("name", "microsoft/trocr-base-handwritten"),
            freeze_encoder=False,
        )

    # Create test dataloader
    loaders = create_dataloaders(config, processor)

    if "test" not in loaders:
        logger.error("Test data not found. Run dataset split first.")
        return {"error": "Test CSV not found."}

    metrics = evaluate_model(
        model=model,
        processor=processor,
        dataloader=loaders["test"],
        device=device,
        num_beams=eval_cfg.get("num_beams", 4),
        max_target_length=model_cfg.get("max_target_length", 32),
    )

    return metrics


if __name__ == "__main__":
    evaluate_from_config()
