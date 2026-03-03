# ScribbleNet

**Transformer-Based Handwritten Word Recognition System**

---

## Overview

ScribbleNet is a production-ready handwritten word recognition system built on the **TrOCR** (Transformer-based Optical Character Recognition) architecture. It fine-tunes `microsoft/trocr-base-handwritten` on the **CVL word-level dataset** to transcribe handwritten word images into digital text.

The system features a complete ML pipeline — from data preprocessing and augmentation, through training and evaluation, to a web-based inference frontend powered by Streamlit.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ScribbleNet                              │
├─────────────┬────────────┬──────────────┬───────────────────────┤
│   DIP       │  Backend   │  Frontend    │  Utilities            │
│ Module      │  Module    │  Module      │  Module               │
├─────────────┼────────────┼──────────────┼───────────────────────┤
│ Grayscale   │ Dataset    │ Streamlit    │ Metrics (WA, CA, ED)  │
│ Denoise     │ Training   │ Upload       │ File Manager          │
│ Threshold   │ Evaluation │ Predict      │ Logger                │
│ Contrast    │ Inference  │ Display      │ Config Loader         │
│ Resize      │ Model I/O  │ Export       │ Structure Validator   │
│ Augment     │            │ (TXT, PDF)   │                       │
└─────────────┴────────────┴──────────────┴───────────────────────┘
         │              │              │
         ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│              TrOCR (Vision Encoder-Decoder)                     │
│  Encoder: ViT (frozen) ──► Decoder: GPT-2 (fine-tuned)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset

**CVL Handwriting Database v1.1**

| Property          | Value              |
|-------------------|--------------------|
| Level             | Word-level         |
| Image format      | TIFF (.tif)        |
| Split strategy    | Writer-based       |
| Train / Val / Test| 80% / 10% / 10%   |

CSV columns: `full_path`, `filename`, `writer_id`, `text_id`, `line_id`, `word_index`, `label`

---

## Model

| Property               | Value                              |
|------------------------|------------------------------------|
| Architecture           | TrOCR (Vision Encoder-Decoder)     |
| Base model             | `microsoft/trocr-base-handwritten` |
| Encoder                | ViT (frozen during fine-tuning)    |
| Decoder                | GPT-2 (fine-tuned)                 |
| Optimizer              | AdamW                              |
| Scheduler              | Linear warmup                      |
| Loss                   | CrossEntropyLoss                   |
| Early stopping         | Patience-based                     |

---

## Project Structure

```
ScribbleNet/
├── data/                  # Dataset files
│   ├── raw/               # Raw CVL database
│   ├── processed/         # Master CSV
│   └── splits/            # Train/Val/Test CSVs
├── models/                # Model artifacts
│   ├── checkpoints/       # Training checkpoints
│   └── exported/          # Best model export
├── backend/               # Core ML modules
│   ├── dataset.py         # PyTorch Dataset & DataLoaders
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation pipeline
│   ├── inference.py       # Inference engine
│   └── model_loader.py    # Model loading & saving
├── dip/                   # Digital Image Processing
│   ├── preprocessing.py   # Preprocessing pipeline
│   └── augmentation.py    # Data augmentation
├── frontend/              # Web interface
│   └── app.py             # Streamlit application
├── config/                # Configuration
│   └── config.yaml        # Central config file
├── utils/                 # Shared utilities
│   ├── metrics.py         # Evaluation metrics
│   ├── file_manager.py    # File & config management
│   └── logger.py          # Logging setup
├── scripts/               # Utility scripts
│   ├── split_dataset.py   # Dataset splitting
│   └── validate_structure.py  # Project validation
├── misc/                  # Relocated redundant files
├── logs/                  # Log files
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── TECHNICAL_DOC.md       # Technical documentation
└── RUN_GUIDE.md           # Setup & run guide
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place CVL dataset in data/raw/cvl-database-1-1/

# 3. Run the CLI menu
python main.py
```

From the menu, select:
- **5** to split the dataset
- **1** to train the model
- **2** to evaluate
- **4** to launch the Streamlit web app

---

## Evaluation Metrics

| Metric                    | Description                                   |
|---------------------------|-----------------------------------------------|
| Word Accuracy (WA)        | Percentage of exactly matched words            |
| Character Accuracy (CA)   | Character-level matching accuracy              |
| Levenshtein Distance (ED) | Average edit operations needed                 |
| Normalized Edit Distance  | Edit distance normalized by string length      |

---

## License

This project is for academic and educational purposes.

---

## Author

Built with ❤️ using PyTorch, HuggingFace Transformers, and Streamlit.
