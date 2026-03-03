"""
ScribbleNet - Metrics Module
Evaluation metrics for handwritten word recognition.
"""

from typing import Dict, List, Tuple

import Levenshtein


def word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute word-level accuracy (exact match percentage).

    Args:
        predictions: List of predicted word strings.
        targets: List of ground truth word strings.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if not predictions:
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
    return correct / len(predictions)


def character_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Compute character-level accuracy across all samples.

    Args:
        predictions: List of predicted word strings.
        targets: List of ground truth word strings.

    Returns:
        Character accuracy as a float between 0 and 1.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if not predictions:
        return 0.0

    total_chars = 0
    correct_chars = 0

    for pred, target in zip(predictions, targets):
        pred = pred.strip()
        target = target.strip()
        total_chars += max(len(pred), len(target))
        for pc, tc in zip(pred, target):
            if pc == tc:
                correct_chars += 1

    return correct_chars / total_chars if total_chars > 0 else 0.0


def levenshtein_distance(predictions: List[str], targets: List[str]) -> float:
    """
    Compute average Levenshtein (edit) distance across all samples.

    Args:
        predictions: List of predicted word strings.
        targets: List of ground truth word strings.

    Returns:
        Average edit distance.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if not predictions:
        return 0.0

    total_dist = sum(
        Levenshtein.distance(p.strip(), t.strip())
        for p, t in zip(predictions, targets)
    )
    return total_dist / len(predictions)


def normalized_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """
    Compute average normalized edit distance (NED).
    NED = edit_distance / max(len(pred), len(target)) per sample.

    Args:
        predictions: List of predicted word strings.
        targets: List of ground truth word strings.

    Returns:
        Average normalized edit distance.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if not predictions:
        return 0.0

    total_ned = 0.0
    for pred, target in zip(predictions, targets):
        pred = pred.strip()
        target = target.strip()
        dist = Levenshtein.distance(pred, target)
        max_len = max(len(pred), len(target))
        total_ned += dist / max_len if max_len > 0 else 0.0

    return total_ned / len(predictions)


def compute_all_metrics(
    predictions: List[str], targets: List[str]
) -> Dict[str, float]:
    """
    Compute all recognition metrics.

    Args:
        predictions: List of predicted word strings.
        targets: List of ground truth word strings.

    Returns:
        Dictionary with all metric values.
    """
    return {
        "word_accuracy": word_accuracy(predictions, targets),
        "character_accuracy": character_accuracy(predictions, targets),
        "avg_levenshtein_distance": levenshtein_distance(predictions, targets),
        "avg_normalized_edit_distance": normalized_edit_distance(predictions, targets),
    }
