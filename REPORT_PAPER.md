# Brain Tumor Segmentation Using Hybrid CNN-Transformer Architecture

**Student:** Musabek Musaev
**Course:** Research Seminar in HW/SW Aspects of Machine Learning for Smart Systems and Communications
**Date:** May 2026

---

## Abstract

This report presents the development and evaluation of a Hybrid U-Net Transformer model for automatic brain tumor segmentation from MRI scans. The project uses the BraTS 2020 dataset containing 369 glioma patient cases. We compare two models: a standard U-Net as baseline and our proposed Hybrid U-Net Transformer. The Hybrid model achieves a mean Dice score of **0.8409** compared to **0.8339** for the baseline U-Net, evaluated on 1,176 validation slices. The system runs on a consumer GPU (NVIDIA RTX 4060, 8 GB VRAM) and completes training in approximately 3–4 hours.

---

## 1. Introduction

Brain tumors are life-threatening conditions that require accurate and fast diagnosis. MRI (Magnetic Resonance Imaging) is the primary tool doctors use to examine brain tumors. However, manually drawing tumor boundaries on MRI images is slow, tiring, and varies between doctors. Automated segmentation using deep learning can solve this problem.

This project addresses two main challenges:
- **Class imbalance** — tumor pixels are only 1–5% of the total image
- **Limited data** — medical datasets are small compared to general computer vision datasets

Our solution combines a classical U-Net (proven in medical imaging) with a Transformer module (capable of understanding the full image context at once).

---

## 2. Dataset

**Dataset:** BraTS 2020 (Brain Tumor Segmentation Challenge 2020)

| Property | Value |
|---|---|
| Total patients | 369 |
| MRI modalities available | T1, T1ce, T2, FLAIR |
| Modalities used | FLAIR + T1ce (2 channels) |
| Segmentation task | Binary — whole tumor vs. background |
| Image resolution | 240 × 240 × 155 voxels (3D) |

**Why FLAIR and T1ce?**
- FLAIR shows the full tumor including surrounding edema
- T1ce highlights the active tumor core with bright contrast

**Data Split:**
- Training: 295 patients (~80%)
- Validation: 74 patients (~20%)
- Split done at **patient level** to prevent data leakage

---

## 3. Data Preprocessing

Since training a full 3D model requires very large GPU memory, we convert 3D MRI volumes into 2D slices:

1. **Load** FLAIR and T1ce volumes for each patient
2. **Normalize** each volume using Z-score normalization (mean and std computed only on brain voxels, not background)
3. **Extract slices** from axial plane, positions 50–130 (outer slices are mostly empty)
4. **Filter** slices that contain fewer than 10 tumor pixels (too empty to be useful)
5. **Crop** each slice to 224 × 224 pixels
6. **Save** as `.npy` files (images and masks separately)

This gives approximately **18,000–22,000 training slices** and **4,500–5,500 validation slices**.

**Data Augmentation** (training only, using Albumentations library):

| Augmentation | Probability |
|---|---|
| Horizontal flip | 50% |
| Vertical flip | 50% |
| Random 90° rotation | 50% |
| Affine (scale, translate, rotate ±15°) | 50% |
| Elastic deformation | 20% |
| Gaussian noise | 20% |
| Brightness/contrast ±10% | 30% |

---

## 4. Model Architecture

### 4.1 Baseline: Standard U-Net

U-Net is the most widely used architecture for medical image segmentation. It consists of:
- **Encoder:** 4 downsampling stages, each with 2 convolutions + MaxPooling. Channels: 32 → 64 → 128 → 256
- **Bottleneck:** 512 channels at 14×14 spatial resolution
- **Decoder:** 4 upsampling stages with skip connections from the encoder
- **Output:** 1×1 convolution producing a binary mask

Total parameters: **7.8 million**

### 4.2 Proposed: Hybrid U-Net Transformer

The key idea is to replace the CNN bottleneck with a Transformer block. This allows the model to "see" the entire image at once and understand global relationships, which helps in detecting tumors that span large or irregular regions.

**Architecture overview:**

```
Input (2 channels, 224×224)
        ↓
  CNN Encoder (4 stages, 32→256 channels)
        ↓
  Patch Embedding (256 channels → 196 tokens)
        ↓
  Transformer (4 blocks, 8 attention heads, d=256)
        ↓
  Project back to spatial (256 channels, 14×14)
        ↓
  CNN Decoder (4 stages, 256→32 channels) + Skip connections
        ↓
  Output head (1×1 conv → binary mask)
```

**Transformer Block details:**
- Multi-Head Self-Attention: 8 heads, head dimension = 32
- Feed-Forward Network: 4× expansion, GELU activation, dropout 0.1
- Pre-LayerNorm design for stable training
- Learnable positional embeddings for the 14×14 = 196 spatial tokens

**Deep Supervision:**
Extra prediction heads are attached at 28×28, 56×56, and 112×112 decoder feature maps. During training, each head contributes to the loss with decreasing weight (1.0, 0.5, 0.25, 0.125). This helps gradients flow better to early layers.

Total parameters: **12.4 million**

---

## 5. Loss Function

Medical image segmentation suffers from severe class imbalance (very few tumor pixels). We use a combined loss:

**Combined Loss = 0.7 × Dice Loss + 0.3 × Focal Loss**

**Dice Loss** directly optimizes the overlap between prediction and ground truth. It handles class imbalance naturally because it focuses on the ratio of correct tumor pixels:

$$L_{Dice} = 1 - \frac{2 \sum p_i y_i + 1}{\sum p_i + \sum y_i + 1}$$

**Focal Loss** reduces the weight of easy background examples and forces the model to focus on hard tumor pixels:

$$L_{Focal} = -\frac{1}{N} \sum \alpha_t (1 - p_t)^{\gamma} \cdot BCE$$

Parameters: γ = 2.0, α = 0.25

---

## 6. Training Setup

| Parameter | U-Net | Hybrid |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 0.0001 | 0.0001 |
| Weight decay | 0.00001 | 0.00001 |
| Batch size | 16 | 8 |
| Max epochs | 50 | 50 |
| LR schedule | Cosine Annealing | Cosine Annealing |
| Warmup epochs | 3 | 3 |
| Early stopping | patience = 10 | patience = 10 |
| Mixed precision | FP16 (AMP) | FP16 (AMP) |
| GPU | RTX 4060 8GB | RTX 4060 8GB |
| Training time | ~2–3 hours | ~3–4 hours |

---

## 7. Evaluation Metrics

Three standard metrics are used:

**Dice Similarity Coefficient (DSC):** Measures overlap between prediction and ground truth. Range 0–1, higher is better.

$$DSC = \frac{2 |P \cap G|}{|P| + |G|}$$

**Intersection over Union (IoU):** Similar to Dice but stricter. Range 0–1, higher is better.

$$IoU = \frac{|P \cap G|}{|P \cup G|}$$

**Hausdorff Distance (HD):** Measures the worst-case boundary error in pixels. Lower is better. Important for clinical use because doctors care about precise tumor boundaries.

---

## 8. Results

### 8.1 Quantitative Results

Evaluated on **1,176 validation slices** from 74 patients:

| Model | Mean Dice ↑ | Median Dice ↑ | Mean IoU ↑ | Mean HD (px) ↓ | Dice Std ↓ |
|---|---|---|---|---|---|
| Baseline U-Net | 0.8339 | 0.9073 | 0.7512 | 14.984 | 0.1993 |
| **Hybrid U-Net Transformer** | **0.8409** | **0.9081** | **0.7577** | **14.266** | **0.1853** |
| Improvement | +0.0070 | +0.0008 | +0.0066 | −0.718 | −0.0140 |

**Key observations:**

- The Hybrid model improves Dice by **+0.70 percentage points**
- Hausdorff Distance improves by **0.72 pixels**, showing better boundary accuracy
- The **standard deviation drops from 0.1993 to 0.1853**, meaning the Hybrid model is more consistent and has fewer completely wrong predictions on hard cases
- Median Dice of 0.907–0.908 shows that most slices are segmented very accurately; the lower mean is caused by a small number of difficult slices

### 8.2 Dice Score Distribution

The distribution of per-slice Dice scores shows that:
- Both models produce high Dice (>0.85) on the majority of slices
- The distribution is left-skewed — a small number of difficult slices (small tumors, irregular shapes) pull the mean down
- The Hybrid model reduces the number of low-Dice outliers, which is reflected in the lower standard deviation

### 8.3 Qualitative Results

Visual comparison of predictions shows that the Hybrid model:
- Produces **smoother, more complete tumor boundaries**
- Creates **fewer isolated false-positive predictions** away from the actual tumor
- Handles **irregular-shaped tumors** better, thanks to the global context from self-attention

---

## 9. Discussion

### Why does the Transformer help?

Convolutional layers only look at small local regions (e.g., 3×3 pixels) at a time. To understand the full tumor extent, information must pass through many layers. The Transformer, on the other hand, connects all 196 spatial positions (14×14 grid) directly in every block. This means it can immediately relate a suspicious region in one corner of the image to another region across the image.

This global reasoning is especially useful for:
- Tumors with large edema (FLAIR hyperintensity spreading over much of the image)
- Irregular tumor shapes that span disconnected regions

### Why does standard deviation decrease?

The reduction in Dice standard deviation (−0.014) is arguably as important as the mean improvement. It means the Hybrid model **fails less catastrophically** on difficult slices. In a clinical setting, consistent predictions are essential — a model that works well on 95% of cases but completely fails on 5% is less useful than one that performs reliably across all cases.

### Hardware efficiency

By using 2D slices instead of 3D volumes, and keeping the Transformer small (4 blocks, d=256), the entire system fits in 8 GB of GPU memory. This is important because it means the method can be used and reproduced without expensive hardware.

---

## 10. Limitations and Future Work

| Limitation | Planned Solution |
|---|---|
| 2D slicing loses 3D context between slices | Train 3D U-Net on higher-memory GPU |
| Binary segmentation only (whole tumor) | Extend to 3-class: edema, tumor core, enhancing tumor |
| No pre-trained weights (trained from scratch) | Use ImageNet pre-trained Transformer encoder |
| Boundary loss not yet activated (weight = 0) | Experiment with boundary loss weight 0.1–0.2 |
| Tested only on BraTS 2020 | Validate on BraTS 2021 and clinical data |
| No test-time augmentation | Apply TTA to improve prediction consistency |

---

## 11. Conclusion

This project successfully implemented and compared two deep learning models for brain tumor segmentation on MRI:

1. **Baseline U-Net:** Mean Dice 0.8339, a solid and well-established architecture
2. **Hybrid U-Net Transformer (proposed):** Mean Dice 0.8409, better IoU, lower Hausdorff Distance, and lower variance

The Transformer bottleneck provides meaningful improvements in segmentation quality, especially for boundary accuracy and robustness on difficult cases. The system is practical, running on a single consumer GPU in under 4 hours, making it accessible for academic research.

The proposed architecture demonstrates that integrating global self-attention into an established CNN framework is an effective and efficient strategy for medical image segmentation under limited data conditions.

---

## References

1. Ronneberger O., Fischer P., Brox T. — *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
2. Dosovitskiy A. et al. — *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
3. Chen J. et al. — *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.* arXiv:2102.04306, 2021.
4. Menze B. H. et al. — *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE Trans. Med. Imaging, 2015.
5. Bakas S. et al. — *Advancing the Cancer Genome Atlas Glioma MRI Collections.* Scientific Data, 2017.
6. Isensee F. et al. — *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation.* Nature Methods, 2021.
7. Milletari F., Navab N., Ahmadi S. — *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV, 2016.
8. Lin T.-Y. et al. — *Focal Loss for Dense Object Detection.* ICCV, 2017.
9. Kervadec H. et al. — *Boundary Loss for Highly Unbalanced Segmentation.* MIDL, 2019.
10. Buslaev A. et al. — *Albumentations: Fast and Flexible Image Augmentations.* Information, 2020.

---

## Appendix: Project File Structure

```
brain_tumor_segmentation/
├── src/
│   ├── config.py           — All hyperparameters
│   ├── preprocess.py       — 3D NIfTI → 2D slices
│   ├── dataset.py          — PyTorch Dataset class
│   ├── transforms.py       — Data augmentation
│   ├── losses.py           — Dice + Focal + Boundary loss
│   ├── metrics.py          — Dice, IoU, Hausdorff
│   ├── train.py            — Training loop
│   ├── evaluate.py         — Evaluation + visualisation
│   ├── report.py           — Comparison plots
│   └── models/
│       ├── unet.py         — Baseline U-Net
│       └── hybrid_unet.py  — Hybrid U-Net Transformer
├── results/
│   ├── comparison.csv      — Final metrics table
│   ├── comparison.png      — Bar chart
│   ├── dice_distribution.png
│   ├── unet_metrics.json
│   └── hybrid_metrics.json
├── checkpoints/


This project was carried out at the University of Klagenfurt,
Study Program: Information and Communication Engineering.

Course: Research Seminar in HW/SW Aspects of Machine Learning for Smart Systems and Communications.

Instructor: Prof. Andrea Tonello.
│   ├── unet_best.pth
│   └── hybrid_best.pth
└── paper_figures/          — All figures for this report
```
