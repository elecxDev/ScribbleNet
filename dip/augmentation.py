"""
ScribbleNet - DIP Augmentation Module
Optional data augmentation for training images.
"""

import random
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def random_rotation(
    image: np.ndarray, angle_range: float = 5.0
) -> np.ndarray:
    """
    Apply a random rotation to the image.

    Args:
        image: Input image.
        angle_range: Maximum rotation angle in degrees (symmetric).

    Returns:
        Rotated image.
    """
    angle = random.uniform(-angle_range, angle_range)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255,
    )
    return rotated


def random_scale(
    image: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)
) -> np.ndarray:
    """
    Apply random scaling to the image.

    Args:
        image: Input image.
        scale_range: Tuple of (min_scale, max_scale).

    Returns:
        Scaled image.
    """
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Ensure output size matches input
    if new_h > h or new_w > w:
        y_start = (new_h - h) // 2
        x_start = (new_w - w) // 2
        scaled = scaled[y_start:y_start + h, x_start:x_start + w]
    else:
        if len(image.shape) == 2:
            canvas = np.ones((h, w), dtype=np.uint8) * 255
        else:
            canvas = np.ones((h, w, image.shape[2]), dtype=np.uint8) * 255
        y_start = (h - new_h) // 2
        x_start = (w - new_w) // 2
        canvas[y_start:y_start + new_h, x_start:x_start + new_w] = scaled
        scaled = canvas

    return scaled


def random_brightness(
    image: np.ndarray, brightness_range: Tuple[float, float] = (0.8, 1.2)
) -> np.ndarray:
    """
    Apply random brightness adjustment.

    Args:
        image: Input image.
        brightness_range: Tuple of (min_factor, max_factor).

    Returns:
        Brightness-adjusted image.
    """
    factor = random.uniform(brightness_range[0], brightness_range[1])
    adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return adjusted


def random_erosion_dilation(
    image: np.ndarray, kernel_size: int = 2, prob: float = 0.3
) -> np.ndarray:
    """
    Randomly apply morphological erosion or dilation to simulate pen thickness variation.

    Args:
        image: Input image (ideally grayscale/binary).
        kernel_size: Size of the morphological kernel.
        prob: Probability of applying the transformation.

    Returns:
        Transformed image.
    """
    if random.random() > prob:
        return image

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if random.random() > 0.5:
        return cv2.erode(image, kernel, iterations=1)
    else:
        return cv2.dilate(image, kernel, iterations=1)


def augment_image(
    image: np.ndarray,
    rotation_range: float = 5.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    apply_morphology: bool = True,
) -> np.ndarray:
    """
    Apply a full augmentation pipeline to an image.

    Args:
        image: Input image.
        rotation_range: Max rotation angle in degrees.
        scale_range: Tuple of (min_scale, max_scale).
        brightness_range: Tuple of (min_brightness, max_brightness).
        apply_morphology: Whether to apply random erosion/dilation.

    Returns:
        Augmented image.
    """
    image = random_rotation(image, angle_range=rotation_range)
    image = random_scale(image, scale_range=scale_range)
    image = random_brightness(image, brightness_range=brightness_range)

    if apply_morphology:
        image = random_erosion_dilation(image)

    return image


def augment_pil_image(
    pil_image: Image.Image,
    rotation_range: float = 5.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    brightness_range: Tuple[float, float] = (0.8, 1.2),
) -> Image.Image:
    """
    Apply augmentation pipeline to a PIL Image.

    Args:
        pil_image: Input PIL Image.
        rotation_range: Max rotation angle in degrees.
        scale_range: Tuple of (min_scale, max_scale).
        brightness_range: Tuple of (min_brightness, max_brightness).

    Returns:
        Augmented PIL Image.
    """
    img_array = np.array(pil_image)
    augmented = augment_image(
        img_array,
        rotation_range=rotation_range,
        scale_range=scale_range,
        brightness_range=brightness_range,
        apply_morphology=False,
    )
    return Image.fromarray(augmented)
