"""
ScribbleNet - DIP Preprocessing Module
Digital Image Processing pipeline for handwritten word images.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a NumPy array (BGR).

    Raises:
        FileNotFoundError: If the image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.

    Args:
        image: Input image (BGR or already grayscale).

    Returns:
        Grayscale image.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def reduce_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.

    Args:
        image: Input image.
        kernel_size: Size of the Gaussian kernel (must be odd).

    Returns:
        Denoised image.
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_adaptive_threshold(
    image: np.ndarray, block_size: int = 11, c: int = 2
) -> np.ndarray:
    """
    Apply adaptive thresholding to binarize the image.

    Args:
        image: Grayscale input image.
        block_size: Size of the neighborhood area.
        c: Constant subtracted from the mean.

    Returns:
        Binary image.
    """
    if len(image.shape) == 3:
        image = to_grayscale(image)
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, c
    )


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Grayscale input image.

    Returns:
        Contrast-enhanced image.
    """
    if len(image.shape) == 3:
        image = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def resize_normalize(
    image: np.ndarray,
    target_height: int = 384,
    target_width: int = 384,
    maintain_aspect: bool = True,
) -> np.ndarray:
    """
    Resize image to target dimensions with optional aspect ratio preservation.

    Args:
        image: Input image.
        target_height: Target height in pixels.
        target_width: Target width in pixels.
        maintain_aspect: If True, maintain aspect ratio and pad.

    Returns:
        Resized image.
    """
    if maintain_aspect:
        h, w = image.shape[:2]
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to target size
        if len(resized.shape) == 2:
            canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
        else:
            canvas = np.ones(
                (target_height, target_width, resized.shape[2]), dtype=np.uint8
            ) * 255

        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas
    else:
        return cv2.resize(
            image, (target_width, target_height), interpolation=cv2.INTER_AREA
        )


def preprocess_pipeline(
    image_path: str,
    grayscale: bool = True,
    noise_reduction: bool = True,
    adaptive_threshold: bool = False,
    contrast_enhancement: bool = True,
    target_size: Optional[Tuple[int, int]] = None,
    gaussian_blur_kernel: int = 3,
) -> Image.Image:
    """
    Run the full DIP preprocessing pipeline on an image.

    Args:
        image_path: Path to the input image.
        grayscale: Whether to convert to grayscale.
        noise_reduction: Whether to apply noise reduction.
        adaptive_threshold: Whether to apply adaptive thresholding.
        contrast_enhancement: Whether to enhance contrast.
        target_size: Optional (height, width) tuple for resizing.
        gaussian_blur_kernel: Kernel size for Gaussian blur.

    Returns:
        Preprocessed image as PIL.Image.
    """
    img = load_image(image_path)

    if grayscale:
        img = to_grayscale(img)

    if contrast_enhancement and len(img.shape) == 2:
        img = enhance_contrast(img)

    if noise_reduction:
        img = reduce_noise(img, kernel_size=gaussian_blur_kernel)

    if adaptive_threshold and len(img.shape) == 2:
        img = apply_adaptive_threshold(img)

    if target_size is not None:
        img = resize_normalize(img, target_size[0], target_size[1])

    # Convert to PIL Image
    if len(img.shape) == 2:
        pil_img = Image.fromarray(img).convert("RGB")
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return pil_img


def preprocess_for_display(
    image_path: str,
    grayscale: bool = True,
    noise_reduction: bool = True,
    contrast_enhancement: bool = True,
    gaussian_blur_kernel: int = 3,
) -> dict:
    """
    Run preprocessing and return intermediate steps for display in frontend.

    Args:
        image_path: Path to the input image.
        grayscale: Whether to convert to grayscale.
        noise_reduction: Whether to apply noise reduction.
        contrast_enhancement: Whether to enhance contrast.
        gaussian_blur_kernel: Kernel size for Gaussian blur.

    Returns:
        Dictionary of step name -> PIL.Image for each processing step.
    """
    steps = {}
    img = load_image(image_path)
    steps["Original"] = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if grayscale:
        img = to_grayscale(img)
        steps["Grayscale"] = Image.fromarray(img).convert("RGB")

    if contrast_enhancement and len(img.shape) == 2:
        img = enhance_contrast(img)
        steps["Contrast Enhanced"] = Image.fromarray(img).convert("RGB")

    if noise_reduction:
        img = reduce_noise(img, kernel_size=gaussian_blur_kernel)
        steps["Noise Reduced"] = Image.fromarray(
            img if len(img.shape) == 3
            else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        )

    thresholded = apply_adaptive_threshold(img) if len(img.shape) == 2 else img
    steps["Adaptive Threshold"] = Image.fromarray(thresholded).convert("RGB")

    return steps
