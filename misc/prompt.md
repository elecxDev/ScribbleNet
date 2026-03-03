You are building a complete, production-ready GitHub repository project named:

ScribbleNet

Transformer-Based Handwritten Word Recognition System

This must be a clean, modular, reproducible, and professional repository suitable for:

Academic presentation

Demonstration

Deployment

Model fine-tuning

Streamlit frontend

GitHub hosting

The repository must NOT become messy.
All code must be modular.
No redundant files.
Unused files must be deleted or moved to /misc automatically.
Structure must be validated at runtime.

🔒 CRITICAL REQUIREMENTS

No hardcoded absolute paths.

Use config file or environment variables.

Automatically create required folders if missing.

If redundant files are detected:

Move to /misc

Or delete safely with confirmation.

Validate directory integrity before running training.

Entire system must run with a single entry point.

Provide Python CLI menu for:

Train model

Evaluate model

Run inference

Launch Streamlit

Clean project

Validate structure

📂 REQUIRED CLEAN PROJECT STRUCTURE
ScribbleNet/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│
├── models/
│   ├── checkpoints/
│   ├── exported/
│
├── backend/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── model_loader.py
│
├── dip/
│   ├── preprocessing.py
│   ├── augmentation.py
│
├── frontend/
│   ├── app.py
│
├── config/
│   ├── config.yaml
│
├── utils/
│   ├── metrics.py
│   ├── file_manager.py
│   ├── logger.py
│
├── scripts/
│   ├── split_dataset.py
│   ├── validate_structure.py
│
├── misc/
│
├── main.py   (CLI menu entry point)
│
├── requirements.txt
├── .gitignore
├── README.md
├── TECHNICAL_DOC.md
├── RUN_GUIDE.md

If any file does not belong in structure:

Move to /misc

📊 DATASET DETAILS

We are using the CVL word-level dataset.

Dataset CSV includes:

full_path

filename

writer_id

text_id

line_id

word_index

label

Implement writer-based 80/10/10 split.

🧠 MODEL REQUIREMENTS

Primary model:

Use:
microsoft/trocr-base-handwritten

Framework:

PyTorch

HuggingFace Transformers

Training:

Freeze encoder layers

Fine-tune decoder

AdamW optimizer

Linear scheduler

CrossEntropy loss

Early stopping

Save best checkpoint

GPU auto-detection

CPU fallback

Evaluation:

Word accuracy

Character accuracy

Levenshtein distance

🖼 DIGITAL IMAGE PROCESSING (DIP) MODULE

Must include preprocessing pipeline:

Grayscale conversion

Noise reduction (Gaussian blur)

Adaptive thresholding

Resize normalization

Contrast enhancement

Optional augmentation (rotation, scaling)

Make DIP modular and optional.

🖥 FRONTEND REQUIREMENTS

Use Streamlit.

Features:

Upload handwritten image

Preview image

Run DIP preprocessing

Run OCR model

Display predicted word

Display confidence score

Show preprocessing steps

Download result as:

TXT

PDF

Clean modern UI

🧩 CLI MENU SYSTEM

main.py must provide interactive menu:

Train model

Evaluate model

Run inference (CLI)

Launch Streamlit app

Split dataset

Validate structure

Clean project

Exit

Must work in one command:

python main.py
📘 DOCUMENTATION REQUIREMENTS

Generate THREE separate documentation files:

README.md

Overview

Project description

Architecture diagram

Dataset info

Model info

Basic usage

TECHNICAL_DOC.md

Model architecture explanation

Transformer explanation

DIP explanation

Training strategy

Evaluation methodology

Design decisions

RUN_GUIDE.md

Setup instructions

Installation steps

Dataset placement

How to train

How to evaluate

How to run frontend

Troubleshooting

Documentation must be clean, professional, and academic-ready.

📦 GIT REQUIREMENTS

Generate:

.gitignore with:

pycache/

*.pyc

data/raw/

models/checkpoints/

models/exported/

.env

venv/

.ipynb_checkpoints/

logs/

Include requirements.txt with all dependencies:

torch

torchvision

transformers

pandas

numpy

scikit-learn

Pillow

opencv-python

streamlit

matplotlib

tqdm

pyyaml

🧠 CODE QUALITY REQUIREMENTS

Modular functions

Type hints

Docstrings

Error handling

Logging

Clean separation of concerns

No redundant duplication

No global variables

Config-driven design

🚀 FINAL GOAL

The repository must be:

Clean

Modular

Professional

Academic-grade

One-command runnable

GitHub ready

Training-ready

Presentation-ready

It must feel like a production AI system.

Build everything end-to-end.