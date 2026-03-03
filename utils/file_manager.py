"""
ScribbleNet - File Manager Module
Handles directory creation, validation, cleanup, and config loading.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from utils.logger import get_logger

logger = get_logger("scribblenet.file_manager")

# Project root is the directory containing this utils package
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config YAML. Defaults to config/config.yaml.

    Returns:
        Dictionary of configuration values.
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config" / "config.yaml")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", config_path)
    return config


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path against the project root.

    Args:
        relative_path: Path relative to project root.

    Returns:
        Absolute Path object.
    """
    return PROJECT_ROOT / relative_path


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create all required directories specified in config.

    Args:
        config: Configuration dictionary with 'paths' section.
    """
    paths = config.get("paths", {})
    for key, rel_path in paths.items():
        full_path = resolve_path(rel_path)
        full_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory: %s", full_path)

    logger.info("All required directories verified.")


def get_expected_structure() -> Dict[str, List[str]]:
    """
    Return the expected project directory/file structure.

    Returns:
        Dictionary mapping directory paths to expected files.
    """
    return {
        "data/raw": [],
        "data/processed": [],
        "data/splits": [],
        "models/checkpoints": [],
        "models/exported": [],
        "backend": [
            "__init__.py", "dataset.py", "train.py",
            "evaluate.py", "inference.py", "model_loader.py",
        ],
        "dip": ["__init__.py", "preprocessing.py", "augmentation.py"],
        "frontend": ["__init__.py", "app.py"],
        "config": ["__init__.py", "config.yaml"],
        "utils": [
            "__init__.py", "metrics.py", "file_manager.py", "logger.py",
        ],
        "scripts": [
            "__init__.py", "split_dataset.py", "validate_structure.py",
        ],
        "misc": [],
        "logs": [],
    }


EXPECTED_ROOT_FILES = [
    "main.py",
    "requirements.txt",
    ".gitignore",
    "README.md",
    "TECHNICAL_DOC.md",
    "RUN_GUIDE.md",
]


def validate_structure(fix: bool = False) -> List[str]:
    """
    Validate that the project structure matches expectations.

    Args:
        fix: If True, create missing directories/files.

    Returns:
        List of issues found.
    """
    issues: List[str] = []
    structure = get_expected_structure()

    for dir_path, expected_files in structure.items():
        full_dir = resolve_path(dir_path)
        if not full_dir.exists():
            issues.append(f"Missing directory: {dir_path}")
            if fix:
                full_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Created missing directory: %s", dir_path)

        for fname in expected_files:
            fpath = full_dir / fname
            if not fpath.exists():
                issues.append(f"Missing file: {dir_path}/{fname}")

    for root_file in EXPECTED_ROOT_FILES:
        if not (PROJECT_ROOT / root_file).exists():
            issues.append(f"Missing root file: {root_file}")

    if not issues:
        logger.info("Project structure validation passed.")
    else:
        logger.warning("Structure issues found: %d", len(issues))

    return issues


def detect_redundant_files() -> List[Path]:
    """
    Detect files in the project root that don't belong to the expected structure.

    Returns:
        List of paths to redundant files.
    """
    expected_dirs = set(get_expected_structure().keys())
    expected_root = set(EXPECTED_ROOT_FILES)
    expected_root.update({
        ".git", ".gitignore", "requirements.txt", "main.py",
        "README.md", "TECHNICAL_DOC.md", "RUN_GUIDE.md",
    })

    redundant: List[Path] = []

    for item in PROJECT_ROOT.iterdir():
        name = item.name
        if name.startswith(".") and name != ".gitignore":
            continue  # skip hidden dirs like .git
        rel = item.relative_to(PROJECT_ROOT).as_posix()
        if item.is_dir():
            if rel not in expected_dirs and name not in {
                "data", "models", ".git", "venv", ".venv", "env",
                "__pycache__",
            }:
                redundant.append(item)
        elif item.is_file():
            if name not in expected_root:
                redundant.append(item)

    return redundant


def clean_project(auto_move: bool = True) -> List[str]:
    """
    Detect redundant files and move them to /misc.

    Args:
        auto_move: If True, automatically moves files. Otherwise, lists them.

    Returns:
        List of actions taken.
    """
    actions: List[str] = []
    misc_dir = resolve_path("misc")
    misc_dir.mkdir(parents=True, exist_ok=True)

    redundant = detect_redundant_files()

    for item in redundant:
        if auto_move:
            dest = misc_dir / item.name
            if dest.exists():
                # Add suffix to avoid overwriting
                stem = dest.stem
                suffix = dest.suffix
                counter = 1
                while dest.exists():
                    dest = misc_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.move(str(item), str(dest))
            action = f"Moved: {item.name} -> misc/{dest.name}"
            logger.info(action)
            actions.append(action)
        else:
            actions.append(f"Redundant: {item.name}")

    if not actions:
        logger.info("No redundant files detected.")
        actions.append("Project is clean. No redundant files found.")

    return actions
