# ScribbleNet — Run Guide

**Step-by-step instructions to set up, train, evaluate, and deploy ScribbleNet.**

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Dataset Placement](#3-dataset-placement)
4. [How to Split the Dataset](#4-how-to-split-the-dataset)
5. [How to Train](#5-how-to-train)
6. [How to Evaluate](#6-how-to-evaluate)
7. [How to Run Inference](#7-how-to-run-inference)
8. [How to Launch the Frontend](#8-how-to-launch-the-frontend)
9. [Configuration Reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9+ | 3.10+ |
| RAM | 8 GB | 16+ GB |
| GPU | Optional | NVIDIA with 6+ GB VRAM |
| Disk space | 5 GB | 10+ GB |
| OS | Windows / Linux / macOS | Any |

---

## 2. Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ScribbleNet.git
cd ScribbleNet
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python main.py
```

You should see the ScribbleNet CLI menu.

---

## 3. Dataset Placement

### CVL Database

1. Download the **CVL Handwriting Database v1.1** from the official source.
2. Extract the contents.
3. Place the dataset folder so the structure is:

```
ScribbleNet/
└── data/
    └── raw/
        └── cvl-database-1-1/
            ├── readme.txt
            ├── testset/
            │   ├── words/
            │   ├── lines/
            │   ├── pages/
            │   └── xml/
            └── trainset/
                ├── words/
                ├── lines/
                ├── pages/
                └── xml/
```

> **Important**: The system reads from `data/raw/cvl-database-1-1/`. If your dataset is elsewhere, update `config/config.yaml` → `dataset.raw_dataset_dir`.

---

## 4. How to Split the Dataset

### Via CLI Menu

```bash
python main.py
# Select option 5: Split Dataset
```

### Via Script Directly

```bash
python -m scripts.split_dataset
```

This will:
1. Scan the raw dataset for word images (`.tif` files)
2. Build a master CSV at `data/processed/cvl_words_dataset.csv`
3. Perform writer-based 80/10/10 split
4. Save to `data/splits/train.csv`, `val.csv`, `test.csv`

---

## 5. How to Train

### Via CLI Menu

```bash
python main.py
# Select option 1: Train Model
```

### What Happens

1. Loads configuration from `config/config.yaml`
2. Loads `microsoft/trocr-base-handwritten` (downloads automatically on first use)
3. Freezes encoder layers
4. Creates DataLoaders from split CSVs
5. Trains with AdamW + linear schedule
6. Validates after each epoch
7. Saves best model to `models/exported/best_model/`
8. Applies early stopping

### Key Training Settings (in `config/config.yaml`)

```yaml
training:
  batch_size: 16       # Reduce if GPU memory is limited
  num_epochs: 25
  learning_rate: 5.0e-5
  early_stopping_patience: 5
  fp16: true           # Set to false if no GPU
```

> **Tip**: For initial testing, set `num_epochs: 2` and `batch_size: 4`.

---

## 6. How to Evaluate

### Via CLI Menu

```bash
python main.py
# Select option 2: Evaluate Model
```

### Via Script Directly

```bash
python -c "from backend.evaluate import evaluate_from_config; evaluate_from_config()"
```

### Output Metrics

- **Word Accuracy**: Exact word matches
- **Character Accuracy**: Character-level correctness
- **Levenshtein Distance**: Average edit distance
- **Normalized Edit Distance**: Length-normalized edit distance

---

## 7. How to Run Inference

### Interactive CLI Mode

```bash
python main.py
# Select option 3: Run Inference (CLI)
```

Then enter image paths one at a time. Type `quit` to exit.

### Programmatic Usage

```python
from backend.inference import ScribbleNetInference
from utils.file_manager import load_config

config = load_config()
engine = ScribbleNetInference(config=config)
result = engine.predict("path/to/image.tif")
print(result["text"], result["confidence"])
```

---

## 8. How to Launch the Frontend

### Via CLI Menu

```bash
python main.py
# Select option 4: Launch Streamlit App
```

### Via Direct Command

```bash
streamlit run frontend/app.py
```

### Frontend Features

- Upload handwritten word images (PNG, JPG, TIFF, BMP)
- Preview uploaded image
- Run OCR with one click
- View predicted text and confidence score
- Toggle DIP preprocessing visualization
- Download results as TXT or PDF

The app opens at `http://localhost:8501` by default.

---

## 9. Configuration Reference

All settings are in `config/config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `paths.raw_data` | | `data/raw` | Raw dataset location |
| `dataset.train_ratio` | | `0.8` | Training split ratio |
| `model.name` | | `microsoft/trocr-base-handwritten` | HuggingFace model ID |
| `model.freeze_encoder` | | `true` | Freeze ViT encoder |
| `training.batch_size` | | `16` | Batch size |
| `training.num_epochs` | | `25` | Max training epochs |
| `training.learning_rate` | | `5e-5` | Initial learning rate |
| `training.fp16` | | `true` | Mixed precision |
| `preprocessing.grayscale` | | `true` | Convert to grayscale |
| `augmentation.enabled` | | `false` | Enable augmentation |

---

## 10. Troubleshooting

### CUDA Out of Memory

Reduce batch size in `config/config.yaml`:

```yaml
training:
  batch_size: 4
```

Or disable FP16 if causing issues:

```yaml
training:
  fp16: false
```

### Model Download Fails

The TrOCR model (~1.2 GB) downloads on first use. Ensure internet access. If behind a proxy:

```bash
set HTTPS_PROXY=http://proxy:port  # Windows
export HTTPS_PROXY=http://proxy:port  # Linux
```

### No Images Found During Split

Ensure the CVL dataset is placed at `data/raw/cvl-database-1-1/` with `trainset/words/` and `testset/words/` subdirectories.

### Streamlit Won't Launch

Ensure Streamlit is installed:

```bash
pip install streamlit
streamlit run frontend/app.py
```

### Import Errors

Run from the project root directory:

```bash
cd ScribbleNet
python main.py
```

### Low Accuracy

- Ensure sufficient training epochs (>10)
- Enable DIP preprocessing
- Check that the dataset split was performed correctly
- Try unfreezing the last few encoder layers

### Windows Path Issues

All paths in ScribbleNet use `pathlib.Path` and `os.path.join` for cross-platform compatibility. If you encounter path issues, ensure you're running from the project root.

---

*ScribbleNet — Run Guide v1.0*
