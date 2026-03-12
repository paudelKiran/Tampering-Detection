# Image Tampering Detection & Localization

A deep learning system for detecting and localizing image tampering using a multi-stream neural network architecture. The model simultaneously analyzes texture, frequency-domain, and noise characteristics of images to produce both pixel-level tampering masks and image-level authenticity predictions.

---

## Architecture Overview

The model employs three parallel feature extraction streams that are fused together for two complementary tasks:

```
                        ┌──────────────────┐
                        │  Input (512×512)  │
                        └──────┬───────────┘
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
       ┌──────────────┐ ┌────────────┐ ┌──────────────┐
       │   Texture     │ │ Frequency  │ │    Noise     │
       │   Stream      │ │  Stream    │ │   Stream     │
       └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
              └───────────────┼───────────────┘
                        ┌─────▼──────┐
                        │  Feature   │
                        │  Fusion    │
                        └──┬─────┬──┘
                           │     │
                    ┌──────▼┐   ┌▼────────────┐
                    │ Seg.  │   │ Classification│
                    │ Head  │   │    Head       │
                    └───────┘   └──────────────┘
```

### Texture Stream
Extracts high-level texture and detail features to capture tampering boundaries and texture inconsistencies.
- Local attention mechanism with a spatial mask
- Multi-scale difference features (coarse-to-fine subtraction)
- Dense refinement blocks
- **Output:** (256, 256, 256) feature maps

### Frequency Stream
Analyses DCT-based frequency-domain characteristics to detect compression artifacts and JPEG incompatibilities.
1. RGB → YCbCr conversion (Y luminance channel)
2. 8×8 block DCT extraction
3. Simulated JPEG quantization (Q = 10)
4. Binary volume encoding across 5 thresholds (−20, −10, 0, 10, 20)
5. HRNet-W18 backbone for multi-scale frequency features

**Preprocessed input shape:** (H/8, W/8, 320) binary volume

### Noise Stream
Isolates sensor noise residuals to detect synthetic insertions and noise pattern inconsistencies.
- **BayarConv** constrained convolution layer (5×5 kernel, center weight fixed to −1, remaining weights normalized to sum to 1) suppresses content and amplifies noise
- ResNet-18 backbone producing 4-scale features:
  - Stage 1: 1/4 resolution, 64 filters
  - Stage 2: 1/8 resolution, 128 filters
  - Stage 3: 1/16 resolution, 256 filters
  - Stage 4: 1/32 resolution, 512 filters

### Feature Fusion
Hierarchical fusion of all three streams via progressive upsampling, concatenation, and channel projection (Conv2D 256 filters).

### Dual Output Heads

| Head | Task | Architecture | Output |
|------|------|-------------|--------|
| **Segmentation** | Pixel-level tampering mask | U-Net style decoder with 4 upsampling levels and multi-scale skip connections | (512, 512, 1) sigmoid |
| **Classification** | Image-level authenticity label | GlobalAveragePooling → Dense(128) → Dropout(0.5) → Dense(1) | (batch, 1) sigmoid |

---

## Loss Functions

| Loss | Formula | Purpose |
|------|---------|---------|
| **Dice Loss** | $1 - \frac{2 \cdot \|A \cap B\| + \epsilon}{\|A\| + \|B\| + \epsilon}$ | Spatial overlap for segmentation |
| **Segmentation Loss** | BCE + Dice | Combines pixel-wise and global structure signals |
| **Classification Loss** | Binary Cross-Entropy | Image-level authenticity supervision |
| **Total Loss** | Segmentation Loss + Classification Loss | Multi-task objective |

---

## Project Structure

```
tamper_model/
├── model.py                # Full model assembly (build_model)
├── train.py                # Training, validation, and evaluation pipeline
├── dataloader.py           # CSV manifest → tf.data pipeline
├── evaluation.py           # Metrics & visualization (TamperingEvaluation)
├── classification.py       # Classification head
├── segmentation.py         # U-Net decoder / segmentation head
├── losses.py               # Dice, BCE, combined loss functions
├── feature_fusion.py       # Multi-stream feature fusion
├── texture_stream.py       # Texture stream with local attention
├── noise_stream.py         # BayarConv + ResNet-18 noise stream
├── frequenct_Stream.py     # DCT preprocessing + frequency branch
├── bayerconv_layer.py      # BayarConv constrained convolution layer
└── hrnet_backbone.py       # HRNet-W18 multi-resolution backbone

manifests/
├── train_manifest.csv      # Training split
├── val_manifest.csv        # Validation split
└── test_manifest.csv       # Test split
```

---

## Data Pipeline

### Manifests

CSV files under `manifests/` define train/val/test splits with the following columns:

| Column | Description |
|--------|-------------|
| `image_path` | Path to the image (.png) |
| `mask_path` | Path to the ground-truth tampering mask (empty for authentic) |
| `json_path` | Path to associated metadata JSON |
| `split` | `train`, `val`, or `test` |
| `label` | `0` = authentic, `1` = tampered |
| `subset` | Category (`authentic` / `tampered`) |
| `fraud_type` | Tampering type (`authentic` for non-tampered) |
| `base_id` | Base image identifier (e.g., `img000001`) |

### Preprocessing
- Images resized to **512 × 512** and normalized to [0, 1]
- Masks resized with nearest-neighbor interpolation and binarized (threshold 127)
- Authentic images receive an all-zeros mask automatically
- tf.data pipeline with shuffle, batch size **8**, and AUTOTUNE prefetching

---

## Evaluation Metrics

The `TamperingEvaluation` class provides:

- **Pixel-level:** Pixel Accuracy, IoU, Dice Coefficient
- **Classification:** Accuracy, Precision, Recall, F1-Score, MCC
- **Visualization:** ROC curves, Confusion matrices

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Input size | 512 × 512 × 3 |
| Batch size | 8 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Epochs | 2 (demo) |
| Classification threshold | 0.5 |

---

## Getting Started

```bash
pip install -r requirements.txt
```

```python
from tamper_model.train import main
main()
```
# The `main()` function will execute the full training and evaluation pipeline, including data loading, model training, and metric reporting. Adjust hyperparameters and paths as needed for your specific dataset and environment.
