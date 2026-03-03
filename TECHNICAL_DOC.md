# ScribbleNet — Technical Documentation

**Transformer-Based Handwritten Word Recognition System**

---

## Table of Contents

1. [Model Architecture](#1-model-architecture)
2. [Transformer Explanation](#2-transformer-explanation)
3. [Digital Image Processing (DIP)](#3-digital-image-processing)
4. [Training Strategy](#4-training-strategy)
5. [Evaluation Methodology](#5-evaluation-methodology)
6. [Design Decisions](#6-design-decisions)

---

## 1. Model Architecture

### TrOCR: Transformer-based OCR

ScribbleNet uses **TrOCR** (Transformer-based Optical Character Recognition), a Vision Encoder-Decoder model that consists of:

- **Encoder**: A Vision Transformer (ViT) that processes the input image into a sequence of patch embeddings.
- **Decoder**: A GPT-2 language model that autoregressively generates the text sequence from the encoder's output.

```
Input Image (384×384 RGB)
        │
        ▼
┌──────────────────────┐
│   Vision Transformer  │  ◄── Frozen during fine-tuning
│   (ViT Encoder)       │
│                        │
│  Image → Patches → Embeddings → Self-Attention
└──────────┬───────────┘
           │ Cross-Attention
           ▼
┌──────────────────────┐
│   GPT-2 Decoder       │  ◄── Fine-tuned
│                        │
│  Encoder output + Previous tokens → Next token prediction
└──────────┬───────────┘
           │
           ▼
      Output Text: "hello"
```

### Base Model

- **Identifier**: `microsoft/trocr-base-handwritten`
- **Pre-training**: The model is pre-trained on printed/handwritten text recognition tasks, making it well-suited for transfer learning on the CVL dataset.
- **Parameters**: ~334M total, with ~200M in the encoder (frozen) and ~134M in the decoder (trainable).

---

## 2. Transformer Explanation

### Self-Attention Mechanism

The core of the Transformer architecture is the **self-attention** mechanism, which allows each position in a sequence to attend to all other positions:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query), $K$ (Key), $V$ (Value) are linear projections of the input
- $d_k$ is the dimension of the key vectors

### Multi-Head Attention

Multiple attention heads capture different types of relationships:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O$$

### Vision Transformer (ViT) Encoder

The ViT encoder processes images by:

1. **Patch Embedding**: Splitting the image into fixed-size patches (16×16) and linearly projecting each patch into an embedding vector.
2. **Positional Encoding**: Adding learnable position embeddings to encode spatial information.
3. **Transformer Blocks**: Passing through multiple layers of multi-head self-attention and feed-forward networks.

### GPT-2 Decoder

The decoder generates text autoregressively:

1. **Cross-Attention**: Attends to encoder outputs to incorporate visual information.
2. **Masked Self-Attention**: Attends only to previously generated tokens (causal masking).
3. **Token Generation**: Produces one token at a time until an end-of-sequence token is generated.

### Beam Search Decoding

At inference time, we use **beam search** with a configurable beam width (default: 4) to find high-probability output sequences:

$$\hat{y} = \arg\max_y \prod_{t=1}^{T} P(y_t | y_{<t}, x)$$

---

## 3. Digital Image Processing

### Preprocessing Pipeline

The DIP module applies a configurable sequence of image transformations to improve recognition quality:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Grayscale Conversion | Reduce color information to single channel |
| 2 | Contrast Enhancement (CLAHE) | Improve visibility of faint strokes |
| 3 | Gaussian Blur | Remove high-frequency noise |
| 4 | Adaptive Thresholding | Binarize to separate text from background |
| 5 | Resize & Normalize | Standardize input dimensions |

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

Unlike global histogram equalization, CLAHE operates on small regions (tiles) and limits contrast amplification to avoid noise amplification:

- **Tile Size**: 8×8 pixels
- **Clip Limit**: 2.0

### Adaptive Thresholding

Uses Gaussian-weighted neighborhood means to compute local thresholds, handling uneven illumination:

$$T(x, y) = \mu(x, y) - C$$

Where $\mu(x,y)$ is the local neighborhood mean and $C$ is a constant.

### Data Augmentation

Optional augmentation techniques to increase training diversity:

| Technique | Range | Purpose |
|-----------|-------|---------|
| Random Rotation | ±5° | Simulate writing angle variation |
| Random Scaling | 0.9×–1.1× | Simulate distance/zoom variation |
| Brightness Jitter | 0.8×–1.2× | Simulate ink/scan variation |
| Morphological Ops | Erosion/Dilation | Simulate pen thickness variation |

---

## 4. Training Strategy

### Transfer Learning with Frozen Encoder

We employ a **partial fine-tuning** strategy:

- **Encoder (ViT)**: Frozen — preserves pre-trained visual feature extraction capabilities.
- **Decoder (GPT-2)**: Fine-tuned — adapts language generation to the CVL vocabulary and writing styles.

This reduces training time, memory requirements, and risk of overfitting while maintaining strong performance.

### Optimizer: AdamW

AdamW decouples weight decay from the adaptive learning rate:

$$\theta_{t+1} = \theta_t - \alpha \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_t\right)$$

- **Learning Rate**: 5e-5
- **Weight Decay**: 0.01
- **Betas**: (0.9, 0.999)

### Learning Rate Schedule

Linear warmup followed by linear decay:

```
LR
│  /\
│ /  \
│/    \────
└───────────── Steps
  warmup  decay
```

- **Warmup Steps**: 500
- **Total Steps**: num_batches × num_epochs

### Early Stopping

Training halts when validation loss fails to improve for a configurable number of epochs (default: 5 epochs patience), preventing overfitting.

### Mixed Precision Training (FP16)

When GPU is available, automatic mixed precision training is used:
- Forward pass in FP16 for speed
- Loss scaling to prevent gradient underflow
- Master weights in FP32 for precision

---

## 5. Evaluation Methodology

### Writer-Based Splitting

The dataset is split by **writer ID** rather than randomly, ensuring:

- No writer appears in multiple splits
- The model is evaluated on truly unseen handwriting styles
- Ratio: 80% train / 10% validation / 10% test

### Metrics

#### Word Accuracy (WA)
$$WA = \frac{\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]}{N}$$

Exact match — the predicted word must be identical to the ground truth.

#### Character Accuracy (CA)
$$CA = \frac{\sum_{i=1}^{N}\sum_{j=1}^{\max(|\hat{y}_i|, |y_i|)} \mathbb{1}[\hat{y}_{ij} = y_{ij}]}{\sum_{i=1}^{N} \max(|\hat{y}_i|, |y_i|)}$$

Character-level matching across all samples.

#### Levenshtein (Edit) Distance
$$ED(a, b) = \min \text{ insertions, deletions, substitutions to transform } a \text{ into } b$$

Average number of single-character edits needed.

#### Normalized Edit Distance (NED)
$$NED = \frac{1}{N}\sum_{i=1}^{N}\frac{ED(\hat{y}_i, y_i)}{\max(|\hat{y}_i|, |y_i|)}$$

Edit distance normalized by the maximum string length per sample.

---

## 6. Design Decisions

### Config-Driven Architecture
All hyperparameters, paths, and settings are centralized in `config/config.yaml`. No hardcoded values exist in any module. This enables:
- Reproducible experiments
- Easy hyperparameter tuning
- Environment-agnostic deployment

### Modular Design
Each module has a single responsibility:
- `backend/` — ML pipeline (dataset, training, evaluation, inference)
- `dip/` — Image processing (preprocessing, augmentation)
- `frontend/` — User interface (Streamlit)
- `utils/` — Shared utilities (metrics, file management, logging)

### Structure Validation
The `file_manager` module validates the project structure at runtime, automatically creating missing directories and detecting/relocating redundant files.

### GPU Auto-Detection with CPU Fallback
The system automatically detects CUDA-capable GPUs and falls back to CPU when unavailable, ensuring portability across environments.

### Writer-Based Splitting
Writer-based splitting prevents data leakage where the model could memorize a writer's style rather than learning to generalize handwriting recognition.

---

*ScribbleNet — Technical Documentation v1.0*
