"""
ScribbleNet - Main Entry Point
Interactive CLI menu for all project operations.

Usage:
    python main.py
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from utils.file_manager import (
    clean_project,
    ensure_directories,
    load_config,
    validate_structure,
)
from utils.logger import setup_logger

BANNER = r"""
 ____            _ _     _     _       _   _      _
/ ___|  ___ _ __(_) |__ | |__ | | ___ | \ | | ___| |_
\___ \ / __| '__| | '_ \| '_ \| |/ _ \|  \| |/ _ \ __|
 ___) | (__| |  | | |_) | |_) | |  __/| |\  |  __/ |_
|____/ \___|_|  |_|_.__/|_.__/|_|\___||_| \_|\___|\__|

   Transformer-Based Handwritten Word Recognition System
                        v1.0.0
"""

MENU = """
╔══════════════════════════════════════════════════╗
║               ScribbleNet - Menu                 ║
╠══════════════════════════════════════════════════╣
║  1. Train Model                                  ║
║  2. Evaluate Model                               ║
║  3. Run Inference (CLI)                          ║
║  4. Launch Streamlit App                         ║
║  5. Split Dataset                                ║
║  6. Validate Structure                           ║
║  7. Clean Project                                ║
║  8. Exit                                         ║
╚══════════════════════════════════════════════════╝
"""


def menu_train() -> None:
    """Launch the training pipeline."""
    print("\n[Training] Starting training pipeline...")
    try:
        from backend.train import train
        results = train()
        if "error" in results:
            print(f"\n  ERROR: {results['error']}")
        else:
            print(f"\n  Training complete!")
            print(f"  Epochs: {results.get('total_epochs', 'N/A')}")
            print(f"  Best Val Loss: {results.get('best_val_loss', 'N/A'):.4f}")
            print(f"  Time: {results.get('total_time_seconds', 0):.1f}s")
    except Exception as e:
        print(f"\n  ERROR: {e}")


def menu_evaluate() -> None:
    """Run model evaluation."""
    print("\n[Evaluation] Starting evaluation...")
    try:
        from backend.evaluate import evaluate_from_config
        metrics = evaluate_from_config()
        if "error" in metrics:
            print(f"\n  ERROR: {metrics['error']}")
        else:
            print("\n  Evaluation Results:")
            print(f"    Word Accuracy:      {metrics.get('word_accuracy', 0):.4f}")
            print(f"    Char Accuracy:      {metrics.get('character_accuracy', 0):.4f}")
            print(f"    Avg Levenshtein:    {metrics.get('avg_levenshtein_distance', 0):.4f}")
            print(f"    Avg NED:            {metrics.get('avg_normalized_edit_distance', 0):.4f}")
            print(f"    Avg Loss:           {metrics.get('avg_loss', 0):.4f}")
    except Exception as e:
        print(f"\n  ERROR: {e}")


def menu_inference() -> None:
    """Run interactive CLI inference."""
    print("\n[Inference] Starting inference mode...")
    try:
        from backend.inference import run_inference_cli
        run_inference_cli()
    except Exception as e:
        print(f"\n  ERROR: {e}")


def menu_streamlit() -> None:
    """Launch the Streamlit frontend."""
    print("\n[Frontend] Launching Streamlit app...")
    app_path = PROJECT_ROOT / "frontend" / "app.py"
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(PROJECT_ROOT),
        )
    except KeyboardInterrupt:
        print("\n  Streamlit app stopped.")
    except Exception as e:
        print(f"\n  ERROR: {e}")


def menu_split() -> None:
    """Run dataset splitting."""
    print("\n[Dataset] Running dataset split...")
    try:
        from scripts.split_dataset import split_dataset
        split_dataset()
    except Exception as e:
        print(f"\n  ERROR: {e}")


def menu_validate() -> None:
    """Validate project structure."""
    print("\n[Validation] Checking project structure...")

    issues = validate_structure(fix=False)

    if issues:
        print(f"\n  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")

        fix = input("\n  Auto-fix missing directories? (y/n): ").strip().lower()
        if fix == "y":
            validate_structure(fix=True)
            print("  Directories fixed.")
    else:
        print("\n  All checks passed! Project structure is valid.")


def menu_clean() -> None:
    """Clean project by moving redundant files to misc."""
    print("\n[Cleanup] Scanning for redundant files...")

    # First show what would be moved
    actions = clean_project(auto_move=False)
    for action in actions:
        print(f"    {action}")

    if any("Redundant" in a for a in actions):
        confirm = input("\n  Move redundant files to /misc? (y/n): ").strip().lower()
        if confirm == "y":
            actions = clean_project(auto_move=True)
            for action in actions:
                print(f"    {action}")
            print("  Cleanup complete.")
        else:
            print("  Cleanup cancelled.")


def main() -> None:
    """Main CLI entry point."""
    print(BANNER)

    # Initialize
    try:
        config = load_config()
        setup_logger(
            name="scribblenet",
            log_file=str(PROJECT_ROOT / config["logging"]["file"]),
            level=config["logging"]["level"],
        )
        ensure_directories(config)
    except Exception as e:
        print(f"  Warning: Could not load config: {e}")
        print("  Running with defaults.\n")

    menu_actions = {
        "1": menu_train,
        "2": menu_evaluate,
        "3": menu_inference,
        "4": menu_streamlit,
        "5": menu_split,
        "6": menu_validate,
        "7": menu_clean,
    }

    while True:
        print(MENU)
        choice = input("  Select option (1-8): ").strip()

        if choice == "8":
            print("\n  Goodbye! 👋\n")
            break
        elif choice in menu_actions:
            menu_actions[choice]()
            input("\n  Press Enter to continue...")
        else:
            print("\n  Invalid option. Please select 1-8.")


if __name__ == "__main__":
    main()
