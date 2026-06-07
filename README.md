# Hybrid AE-MSCNN for Binary Zero-Day Network Intrusion Detection on NSL-KDD

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)](https://pytorch.org)
[![NSL-KDD](https://img.shields.io/badge/Dataset-NSL--KDD-green)](https://www.unb.ca/cic/datasets/nsl.html)

A hybrid deep learning architecture that combines an **Autoencoder (AE)** with a **Multi-Scale 1D CNN (MSCNN)** and a **learned MLP calibrator** for detecting zero-day (unknown) network intrusions. The system is evaluated on the **NSL-KDD** benchmark, where 17 attack types appear only in the test set and are never seen during training.

---

## Architecture

The system fuses three components in a two-branch design.

### 1. Autoencoder Branch (Unsupervised Anomaly Detection)

The first branch is trained only on normal (benign) traffic. Its job is to learn a compact representation of legitimate behavior.

```
Encoder:  input_dim -> 256 -> 128 -> latent_dim (64)
Decoder:  latent_dim (64) -> 128 -> 256 -> input_dim
```

- **Loss:** MSE reconstruction error
- **Latent Space:** 64-dimensional bottleneck
- **Anomaly Scoring:** Reconstruction error is converted to a **q-score**, defined as the percentile of the per-sample error relative to the normal validation distribution
- **Dropout:** 0.2, trained with AdamW (lr=1e-3, weight_decay=1e-5)
- **Early stopping** on validation MSE (patience=30)

### 2. Multi-Scale 1D CNN Branch (Supervised Classification)

The second branch is a discriminative classifier that processes a **308-dimensional fused input** (see [Feature Pipeline](#feature-pipeline)) through parallel convolutions at three different scales.

```
Linear Stem: 308 -> 64 (BN + ReLU) -> reshape to (B, 1, 64)

Multi-Scale Branches (parallel, each 1 -> 32 channels):
  +-- Conv1D(k=3, pad=1) + BN + ReLU   (Narrow)
  +-- Conv1D(k=5, pad=2) + BN + ReLU   (Medium)
  +-- Conv1D(k=7, pad=3) + BN + ReLU   (Wide)

Merge: Conv1D(96 -> 96, k=1) + BN + ReLU
  -> Residual Block (96 -> 128)
  -> MaxPool1D(k=2)
  -> Residual Block (128 -> 128)

Each Residual Block:
  Conv1D(k=3) -> BN -> ReLU -> Dropout(0.25)
  -> Conv1D(k=3) -> BN -> SqueezeExcite1D -> ReLU (+ shortcut)

Global Pooling: Concat(AdaptiveMaxPool1d, AdaptiveAvgPool1d) -> 256
Head: Linear(256 -> 128) -> ReLU -> Dropout -> Linear(128 -> 2)
```

- **Squeeze-and-Excitation** attention recalibrates channel-wise feature responses
- **Kernel sizes:** 3, 5, 7 to capture patterns at multiple temporal scales
- **Optimizer:** AdamW (lr=8e-4), trained for up to 30 epochs with early stopping

### 3. MLP Calibrator (Learned Fusion)

The final component is a lightweight neural gate that fuses CNN and AE signals into a unified decision.

```
Input (7+ features) -> Linear(-> 32) -> BN -> ReLU -> Dropout(0.3) -> Linear(-> 1) -> Sigmoid
```

**Calibrator input features:**
| Feature | Source | Description |
|---|---|---|
| `cnn_attack_prob` | CNN | Temperature-scaled softmax probability of the attack class |
| `ae_q_score` | AE | Percentile of reconstruction error compared to normal validation |
| `cnn_entropy` | CNN | Binary entropy of CNN prediction |
| `recon_mse_z` | AE | Standardized per-sample MSE |
| `recon_mae_z` | AE | Standardized per-sample MAE |
| `recon_max_abs_z` | AE | Standardized per-sample max absolute residual |
| `recon_weighted_mse_z` | AE | Feature-weighted MSE (W = 1/sigma^2) |
| Residual PCA | AE | Up to 16 PCA components on squared residuals (optional) |

- Trained using **weighted BCE loss** (class-balanced)
- **Constraint-driven optimization:** selects the decision threshold that maximizes attack recall while keeping FPR <= 0.12
- **Optimizer:** AdamW with weight_decay = 1.0 (inverse regularization)

---

## Feature Pipeline

```
Raw Input (122-dim after OHE + scaling)
        |
        +---> Autoencoder (normal-only) -> latent (64) + reconstruction
        |       +-- residual_vector = |X - X_recon|^2  (122-dim)
        |
        +---> Fused Vector = [raw_122; latent_64; residual_122]  -> 308-dim
                        |
                        +---> Multi-Scale 1D CNN -> softmax probabilities
                              + AE q-score + entropy + recon stats
                        |
                        +---> MLP Calibrator -> final binary decision
```

---

## Data Preprocessing and Balancing

- **Numerical features** (38 dim): Standard Z-score scaling
- **Categorical features** (3: protocol_type, service, flag): One-Hot Encoding -> 84 dim
- **Total base dimension:** 122
- **Class imbalance:** mitigated using **Borderline-SMOTE** (borderline-1 variant) applied to the CNN training split. This generates synthetic samples near the decision boundary without introducing artificial noise in stable regions.
- **Validation split:** 15% stratified holdout from training
- **Evaluation protocol:** strict NSL-KDD split: train on **KDDTrain+**, test on **KDDTest+**

---

## Experimental Results

### Overall Performance

| Metric | Standalone CNN | Hybrid AE-MSCNN + Calibrator |
|---|---|---|
| **Overall Accuracy** | 78.14% | **88.44%** |
| **Balanced Accuracy** | 80.39% | **88.50%** |
| **Macro F1-Score** | 78.08% | **88.28%** |
| **Attack Precision** | 96.25% | 91.31% |
| **Attack Recall** | 64.08% | **88.08%** |
| **Attack F1-Score** | 76.94% | **89.66%** |
| **ROC-AUC** | 0.8828 | **0.9572** |
| **PR-AUC** | 0.9127 | **0.9635** |
| **Normal FPR** | 3.30% | 11.08% |

### Per-Class Breakdown

| Subset | Standalone CNN | Hybrid AE-MSCNN |
|---|---|---|
| **Known Subset Accuracy** | 86.56% | 89.43% |
| **Unknown Subset Accuracy** | 35.89% | **83.47%** |
| **Known Attack Recall** | 75.72% | 89.98% |
| **Unknown Attack Recall** | 35.89% | **83.47%** |

The hybrid architecture neutralized **13 of 17** unknown attack types, more than doubling the zero-day detection rate of the standalone CNN baseline.

### Confusion Matrices (Strict Split)

| | Standalone CNN | Hybrid AE-MSCNN |
|---|---|---|
| **TN / FP** | 9,391 / 320 | 8,635 / 1,076 |
| **FN / TP** | 4,609 / 8,224 | 1,530 / 11,303 |

The hybrid system trades a modest increase in the false positive rate (3.3% to 11.1%) for a large improvement in unknown attack recall (35.9% to 83.5%).

### Closed-World Performance (Merged Split 80/20)

On the merged random split, where no zero-day separation exists, the CNN alone already performs well and the hybrid calibrator further boosts recall at the cost of a slightly higher FPR.

| Metric | Standalone CNN | Hybrid AE-MSCNN + Calibrator |
|---|---|---|
| **Overall Accuracy** | 97.10% | 95.06% |
| **Balanced Accuracy** | 97.20% | 95.23% |
| **Macro F1-Score** | 97.10% | 95.06% |
| **Attack Precision** | 94.59% | 90.76% |
| **Attack Recall** | 99.69% | **99.90%** |
| **Attack F1-Score** | 97.07% | 95.11% |
| **ROC-AUC** | 0.9915 | **0.9990** |
| **PR-AUC** | 0.9756 | **0.9983** |
| **Normal FPR** | 5.29% | 9.43% |

---

## Repository Structure

```
+-- organized-ae-cnn-main-pipeline.ipynb     Main pipeline notebook (PyTorch)
+-- nslkdd_dataset_report.py                 Dataset analysis and visualization script
+-- nslkdd_dataset_report/                   Generated dataset shift report
|   +-- report.md                            Auto-generated report
|   +-- summary.json                         Dataset summary
|   +-- *.csv                                Feature shift tables
|   +-- *.png                                Visualization plots
+-- NSL-KDD/                                 Raw NSL-KDD dataset files
|   +-- KDDTrain+.txt / KDDTest+.txt         Standard splits
|   +-- KDDTrain+_20Percent.txt              Subset for fast experimentation
|   +-- *.arff                               ARFF format versions
|   +-- nsl-kdd/                             Kaggle mirror
+-- Presentations/                           Project presentations (3 phases)
|   +-- IDS-Phase-1.pdf
|   +-- IDS-Phase-2.pdf
|   +-- IDS-Phase-3.pdf
+-- results/                                 Training outputs and run artifacts
|   +-- kaggle/working/training_runs/
+-- Full Report - Project ML - Oussama Ben Sassi.pdf
+-- README.md
```

---

## Usage

### 1. Dataset

The NSL-KDD dataset files are included in the `NSL-KDD/` directory. You can also download them from [Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd).

### 2. Run the Pipeline

The main pipeline is in `organized-ae-cnn-main-pipeline.ipynb`. It supports two evaluation modes:

- **Strict split** (default): train on KDDTrain+, test on KDDTest+. This is the zero-day detection scenario.
- **Merged split:** 80/20 random train/test split for closed-world evaluation.

All key parameters are configurable at the top of the notebook.

### 3. Dataset Report

```bash
python nslkdd_dataset_report.py --dataset-dir NSL-KDD/ --output-dir nslkdd_dataset_report
```

### 4. Requirements

- Python 3.12
- PyTorch 2.0
- scikit-learn, imbalanced-learn
- pandas, numpy, matplotlib, seaborn

---

## Tech Stack

- **Framework:** PyTorch 2.0
- **Language:** Python 3.12
- **Imbalance handling:** imbalanced-learn (Borderline-SMOTE)
- **Visualization:** Matplotlib, Seaborn
- **Hardware:** Optimized for NVIDIA T4 GPU (Kaggle environment)


