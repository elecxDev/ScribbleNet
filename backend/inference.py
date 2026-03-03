"""
ScribbleNet - Inference Module
Run handwritten word recognition on single images or batches.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from dip.preprocessing import preprocess_pipeline
from utils.logger import get_logger

logger = get_logger("scribblenet.inference")


class ScribbleNetInference:
    """
    Inference engine for ScribbleNet handwritten word recognition.
    """

    def __init__(
        self,
        model: Optional[VisionEncoderDecoderModel] = None,
        processor: Optional[TrOCRProcessor] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the inference engine.

        Args:
            model: Pre-loaded model. If None, loads from config.
            processor: Pre-loaded processor. If None, loads from config.
            device: Target device. Auto-detected if None.
            config: Configuration dictionary.
        """
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            self.device = device or torch.device("cpu")
        elif config is not None:
            self._load_from_config(config)
        else:
            raise ValueError("Provide either (model, processor) or config.")

        self.model.eval()
        self.config = config or {}
        logger.info("Inference engine initialized.")

    def _load_from_config(self, config: Dict[str, Any]) -> None:
        """Load model from configuration."""
        from backend.model_loader import load_checkpoint, load_model_and_processor
        from utils.file_manager import resolve_path

        exported_dir = resolve_path(config["paths"]["exported_dir"])
        best_model_dir = exported_dir / "best_model"

        if best_model_dir.exists():
            logger.info("Loading fine-tuned model from: %s", best_model_dir)
            self.model, self.processor, self.device = load_checkpoint(
                str(best_model_dir)
            )
        else:
            model_cfg = config.get("model", {})
            logger.info("Loading base model: %s", model_cfg.get("name"))
            self.model, self.processor, self.device = load_model_and_processor(
                model_name=model_cfg.get("name", "microsoft/trocr-base-handwritten"),
                freeze_encoder=False,
            )

    def predict(
        self,
        image: Union[str, Path, Image.Image],
        apply_dip: bool = True,
        num_beams: int = 4,
        max_length: int = 32,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image: Image path, Path object, or PIL Image.
            apply_dip: Whether to apply DIP preprocessing.
            num_beams: Number of beams for beam search.
            max_length: Maximum generated sequence length.

        Returns:
            Dictionary with 'text' and 'confidence' keys.
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image_path = str(image)
            if apply_dip:
                preprocess_cfg = self.config.get("preprocessing", {})
                pil_image = preprocess_pipeline(
                    image_path,
                    grayscale=preprocess_cfg.get("grayscale", True),
                    noise_reduction=preprocess_cfg.get("noise_reduction", True),
                    contrast_enhancement=preprocess_cfg.get("contrast_enhancement", True),
                    gaussian_blur_kernel=preprocess_cfg.get("gaussian_blur_kernel", 3),
                )
            else:
                pil_image = Image.open(image_path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Process image
        pixel_values = self.processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values.to(self.device)

        # Generate prediction with scores
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode
        generated_ids = outputs.sequences
        predicted_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Compute confidence score (average token probability)
        confidence = self._compute_confidence(outputs)

        result = {
            "text": predicted_text,
            "confidence": confidence,
        }

        logger.info(
            "Prediction: '%s' (confidence: %.2f%%)",
            predicted_text, confidence * 100,
        )

        return result

    def _compute_confidence(self, outputs: Any) -> float:
        """
        Compute confidence score from generation outputs.

        Args:
            outputs: Model generation outputs with scores.

        Returns:
            Confidence score between 0 and 1.
        """
        if not hasattr(outputs, "scores") or not outputs.scores:
            return 0.0

        scores = outputs.scores
        probs = [torch.softmax(score, dim=-1).max(dim=-1).values for score in scores]
        avg_confidence = torch.stack(probs).mean().item()
        return avg_confidence

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        apply_dip: bool = True,
        num_beams: int = 4,
        max_length: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of image paths or PIL Images.
            apply_dip: Whether to apply DIP preprocessing.
            num_beams: Number of beams for beam search.
            max_length: Maximum generated sequence length.

        Returns:
            List of result dictionaries.
        """
        results = []
        for image in images:
            try:
                result = self.predict(
                    image, apply_dip=apply_dip,
                    num_beams=num_beams, max_length=max_length,
                )
                results.append(result)
            except Exception as e:
                logger.error("Error processing image %s: %s", image, e)
                results.append({"text": "", "confidence": 0.0, "error": str(e)})
        return results


def run_inference_cli(config_path: Optional[str] = None) -> None:
    """
    Interactive CLI inference mode.

    Args:
        config_path: Path to configuration YAML.
    """
    from utils.file_manager import load_config

    config = load_config(config_path)
    engine = ScribbleNetInference(config=config)

    print("\n" + "=" * 50)
    print("  ScribbleNet - Inference Mode")
    print("=" * 50)
    print("Enter image path (or 'quit' to exit):\n")

    while True:
        image_path = input("Image path: ").strip()
        if image_path.lower() in ("quit", "exit", "q"):
            break
        if not Path(image_path).exists():
            print(f"  File not found: {image_path}")
            continue

        result = engine.predict(image_path)
        print(f"  Predicted: {result['text']}")
        print(f"  Confidence: {result['confidence'] * 100:.2f}%")
        print()


if __name__ == "__main__":
    run_inference_cli()
