# GRAIL Plant Segmentation — Machine Vision & Processing Report

## Project Overview

The GRAIL (Greenhouse Robotics and AI Lab) project developed a real-time semantic segmentation system for identifying plant structures in greenhouse images. The system classifies every pixel in a camera frame into one of three classes: **Background**, **Leaf**, or **Stem/Petiole**. It is designed for deployment on a **Raspberry Pi 5 + OAK-D Pro** depth camera for downstream robotic grasping tasks.

- **GitHub**: https://github.com/ethancg27-oss/grail_tamucc.git
- **Framework**: PyTorch + torchvision
- **Deployment target**: Intel MyriadX VPU (OAK-D Pro camera) via DepthAI

---

## 1. Dataset

- **Source**: [Synthetic Plants](https://datasetninja.com/synthetic-plants) from Dataset Ninja
- **Format**: Supervisely (JSON annotations with bitmap/polygon geometries)
- **Split**: 8,000 train / 1,000 val / 1,000 test images
- **Classes**: Background (0), Leaf (1), Stem/Petiole (2)

### Data Pipeline (`scripts/datasets/synthetic_plants.py`)

The dataset adapter converts raw Supervisely-format data into a canonical training format:

1. Auto-detects the annotation format (Supervisely JSON, COCO, or pre-rendered machine masks)
2. Parses bitmap and polygon annotations from Supervisely JSONs (base64 → zlib → PNG decoding for bitmaps)
3. Remaps source labels (`leaf` → 1, `petiole`/`stem`/`branch` → 2, everything else → 0)
4. Resizes images and masks to the target resolution
5. Generates train/val/test split files

---

## 2. Model Architecture

### LRASPP-MobileNetV3-Large

The selected architecture is **LRASPP** (Lite Reduced Atrous Spatial Pyramid Pooling) with a **MobileNetV3-Large** backbone. This was chosen for its balance of accuracy and inference speed on edge hardware.

- **Backbone**: MobileNetV3-Large (ImageNet-pretrained)
- **Decoder**: LRASPP — a lightweight segmentation head that uses a low-resolution branch (40 channels) and a high-resolution branch (128 channels)
- **Output**: 3-channel logit map (one per class) at the input resolution
- **Transfer learning**: The pretrained backbone was kept; only the final classification convolutions (low_classifier and high_classifier) were replaced for 3-class output

### Why LRASPP over DeepLabV3?

Both LRASPP and DeepLabV3 with MobileNetV3-Large were evaluated (runs 18–19 vs. run 21). LRASPP significantly outperformed DeepLabV3 on this dataset:

| Architecture | Best mIoU |
|---|---|
| LRASPP | 0.714 |
| DeepLabV3 | 0.630 |

DeepLabV3's richer ASPP decoder added parameters without benefit on these relatively simple 3-class images.

---

## 3. Training Pipeline (`scripts/train_segmentation.py`)

### Preprocessing
- Images resized to target resolution (256–768 px square)
- **ImageNet normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- HWC → CHW tensor conversion

### Data Augmentation

Two augmentation levels were implemented:

**Basic augmentation** (used in best runs):
- Random horizontal/vertical flip
- Random 90-degree rotation
- Random brightness/contrast adjustment
- Random HSV color jitter

**Strong augmentation** (tested, did not improve results):
- All basic augmentations plus:
- Continuous rotation (-45 to +45 degrees)
- Random scale and crop (0.8x–1.2x)
- Elastic deformation (for thin structures like stems)
- Gaussian blur and noise
- Random erasing / cutout

### Loss Functions Explored

Six loss functions were implemented and compared:

| Loss | Description | Result |
|---|---|---|
| **Cross-Entropy (CE)** | Standard pixel-wise classification loss | Baseline |
| **CE + Dice** | CE for stable gradients + Dice for per-class overlap | Best at 256px |
| **CE + Tversky (alpha=0.3, beta=0.7)** | Asymmetric FN/FP penalty — penalizes missed stems more | **Best overall at 512px+** |
| **CE + Lovasz** | CE + Lovasz-Softmax (direct IoU surrogate) | Competitive but not best |
| **Pure Lovasz** | Lovasz-Softmax alone | **Diverged** (run 17, mIoU=0.288) |
| **Focal Loss** | Down-weights easy pixels | Available but not top performer |

The **Tversky loss** was key to stem performance. With alpha=0.7 (FN penalty) and beta=0.3 (FP penalty), it asymmetrically punished the model for missing stem pixels more than for false-positive stems — a critical property for thin, hard-to-segment structures.

### Class Weighting

Loss weights of **0.5 / 2.0 / 3.0** (BG / Leaf / Stem) addressed the severe class imbalance where background dominates most images and stems occupy very few pixels.

### Stem Sampling

A **WeightedRandomSampler** oversampled training images containing more stem pixels, providing a modest ~1% mIoU improvement.

### Optimizer and Scheduler
- **Optimizer**: AdamW
- **Learning rate**: 1e-3
- **Gradient clipping**: max_norm=1.0
- **Scheduler**: CosineAnnealingLR

### Metrics
- **Per-class IoU** (Intersection over Union) computed from a cumulative confusion matrix
- **Mean IoU** (mIoU) across all 3 classes — the primary metric for model selection
- Training curves (loss, mIoU) saved automatically per run

---

## 4. Experimental Results — 33 Training Runs

### Full Run Table

| Run | Model | Loss | Aug | Class Wts (BG/LF/ST) | Epochs | Res | BG IoU | Leaf IoU | Stem IoU | mIoU |
|-----|-------|------|-----|-----------------------|--------|-----|--------|----------|----------|------|
| 01–07 | LRASPP | CE | - | various | 50–100 | 256 | - | - | - | - |
| 08* | LRASPP | CE+Dice | - | 0.5/2.0/3.0 | 100 | 256 | 0.8865 | 0.6750 | 0.5541 | 0.7135 |
| 09 | LRASPP | CE | - | 0.5/2.0/3.0 | 100 | 256 | 0.8817 | 0.6793 | 0.5260 | 0.6957 |
| 12 | LRASPP | CE+Dice | - | 0.5/2.0/3.0 | 100 | 256 | 0.8866 | 0.6764 | 0.5437 | 0.7022 |
| 13 | LRASPP | CE+Dice | - | 0.5/2.0/3.0 | 100 | 256 | 0.8914 | 0.6808 | 0.5482 | 0.7068 |
| 14 | LRASPP | CE+Dice | strong | 0.5/2.0/3.0 | 250 | 256 | 0.8854 | 0.6717 | 0.5310 | 0.6960 |
| 15 | LRASPP | CE+Dice | strong | 0.5/2.0/3.0 | 250 | 256 | 0.8833 | 0.6694 | 0.5267 | 0.6931 |
| 16 | LRASPP | CE+Dice | strong | 0.5/2.0/3.0 | 200 | 256 | 0.8750 | 0.6614 | 0.5118 | 0.6827 |
| 17 | LRASPP | Lovasz | basic | 0.5/2.0/3.0 | 200 | 256 | 0.7441 | 0.0991 | 0.0221 | 0.2884 |
| 18 | DeepLabV3 | CE+Dice | basic | 0.5/2.0/3.0 | 200 | 256 | 0.8417 | 0.6040 | 0.4404 | 0.6287 |
| 19 | DeepLabV3 | CE+Lovasz | basic | 0.5/2.0/3.0 | 71+ | 256 | 0.8577 | 0.5723 | 0.3846 | 0.6049 |
| 20 | LRASPP | CE+Tversky | basic | 0.5/2.0/3.0 | 100 | 256 | 0.8865 | 0.6750 | 0.5441 | 0.7019 |
| 21+ | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 256 | 0.8953 | 0.6903 | 0.5559 | 0.7139 |
| 22 | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 256 | 0.8927 | 0.6926 | 0.5513 | 0.7122 |
| 23 | LRASPP | CE+Lovasz | basic | 0.5/2.0/3.0 | 100 | 256 | 0.9070 | 0.6935 | 0.5010 | 0.7005 |
| 24 | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 320 | 0.9099 | 0.7235 | 0.6066 | 0.7467 |
| 25 | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 384 | 0.9224 | 0.7554 | 0.6533 | 0.7770 |
| 26 | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 256 | 0.8913 | 0.6858 | 0.5490 | 0.7087 |
| 27 | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 256 | 0.8906 | 0.6851 | 0.5435 | 0.7064 |
| 28 | LRASPP | CE+Lovasz | basic | 0.5/2.0/3.0 | 100 | 256 | 0.9047 | 0.6907 | 0.4932 | 0.6962 |
| 29 | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 448 | 0.9295 | 0.7718 | 0.6781 | 0.7931 |
| 30 | LRASPP | CE+Dice | basic | 0.5/2.0/3.0 | 100 | 512 | 0.9380 | 0.7906 | 0.6810 | 0.7950 |
| 31 | LRASPP | CE+Tversky | basic | 0.5/2.0/3.0 | 150 | 512 | 0.9447 | 0.8040 | 0.7382 | 0.8290 |
| 32 | LRASPP | CE+Tversky | basic | 0.5/2.0/3.0 | 150 | 640 | 0.9505 | 0.8224 | 0.7652 | 0.8460 |
| **33** | **LRASPP** | **CE+Tversky** | **basic** | **0.5/2.0/3.0** | **150** | **768** | **0.9574** | **0.8410** | **0.7964** | **0.8649** |

### Key Findings

1. **Resolution is the #1 factor**: Scaling from 256px to 768px yielded a **+0.151 mIoU** improvement. Every resolution step produced consistent gains.
2. **CE+Tversky dominates at high resolution**: At 256px, CE+Dice was best. But at 512px+, CE+Tversky pulled ahead by a wide margin, especially on stem IoU.
3. **Strong augmentation hurts**: Runs 14–16 with strong augmentation consistently underperformed basic augmentation, likely because the synthetic data was already diverse enough.
4. **Pure Lovasz diverges**: Run 17 showed that Lovasz-Softmax alone is unstable without a CE anchor for early-training gradients.
5. **LRASPP >> DeepLabV3**: The lightweight decoder outperformed the heavier one on this dataset (0.714 vs 0.630 mIoU).

### Resolution Sweep (Best at Each Resolution)

| Resolution | Run | Loss | Best mIoU | Stem IoU |
|------------|-----|------|-----------|----------|
| 256 | run21 | CE+Dice | 0.714 | 0.557 |
| 320 | run24 | CE+Dice | 0.746 | 0.603 |
| 384 | run25 | CE+Dice | 0.777 | 0.653 |
| 448 | run29 | CE+Dice | 0.793 | 0.678 |
| 512 | run31 | CE+Tversky | 0.829 | 0.738 |
| 640 | run32 | CE+Tversky | 0.846 | 0.765 |
| 768 | run33 | CE+Tversky | 0.865 | 0.796 |

### Best Model — Run 33 Performance

- **Background IoU**: 95.7%
- **Leaf IoU**: 84.1%
- **Stem IoU**: 79.6% (exceeds the 70% project target)
- **Mean IoU**: 86.49%
- **Recall**: BG 96.9%, Leaf 94.0%, Stem 93.0%
- **Precision**: BG 98.7%, Leaf 88.9%, Stem 84.7%

The lower precision vs. recall for stems (84.7% vs 93.0%) means the model slightly over-predicts stem regions — a preferable failure mode for robotic grasping, where missing a stem is worse than slightly overestimating its boundary.

---

## 5. Edge Deployment — Export Pipeline

The deployment target is the **Intel MyriadX VPU** inside the OAK-D Pro camera, which requires a compiled `.blob` file.

### Pipeline: PyTorch → ONNX → OpenVINO IR → DepthAI Blob

#### Step 1: HardSigmoid/Hardswish Workaround

MyriadX (OpenVINO 2022.1) does not support `HardSigmoid` and `Hardswish` activation ops natively. Before ONNX export, these are replaced at the module level:

- `nn.Hardsigmoid` → `relu6(x + 3) / 6`
- `nn.Hardswish` → `x * relu6(x + 3) / 6`

#### Step 2: ONNX Export

```python
torch.onnx.export(
    wrapper, dummy_input, output_path,
    opset_version=11,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None,  # Fixed shape for OAK deployment
)
```

- **Opset 11** required for MyriadX compatibility
- The `SegmentationModelWrapper` class unwraps the torchvision dict output (`{'out': tensor}`) into a plain tensor for a clean single-output ONNX graph
- Fixed input shape (no dynamic axes) for deterministic VPU compilation

#### Step 3: Blob Compilation (`scripts/compile_768_norm.py`)

```python
blobconverter.from_onnx(
    model='exports/plant_seg_lraspp_opset11.onnx',
    data_type='FP16',
    shaves=6,
    optimizer_params=[
        '--reverse_input_channels',
        '--mean_values=[123.675,116.28,103.53]',
        '--scale_values=[58.395,57.12,57.375]',
        '--input_shape=[1,3,768,768]',
    ],
)
```

Key details:
- **FP16** precision for MyriadX inference
- **6 shaves** (VPU compute units)
- **ImageNet normalization baked in** via `--mean_values` and `--scale_values` so the camera feed can be passed in as raw uint8 BGR without any preprocessing on the Pi
- **`--reverse_input_channels`** handles BGR→RGB conversion inside the blob
- Compilation done via blobconverter's cloud service (uses OpenVINO 2022.1's `mo` — local `mo` version mismatch caused failures)

### Exported Artifacts

| File | Resolution | Size | Description |
|------|-----------|------|-------------|
| `exports/plant_seg_lraspp.blob` | 768px | 7.8 MB | Best model (run33), normalization baked in |
| `exports/plant_seg_lraspp_640.blob` | 640px | — | Fallback (run32), normalization baked in |
| `exports/plant_seg_lraspp_opset11.onnx` | 768px | — | ONNX (HardSigmoid decomposed) |
| `exports/plant_seg_lraspp_640_opset11.onnx` | 640px | — | ONNX (HardSigmoid decomposed) |

---

## 6. Real-World Validation — Microgreen Demo

After training on synthetic data, the model was tested on **real photographs of microgreen plants** growing in white trays under greenhouse lights. This validated domain transfer from synthetic to real-world imagery.

### Inference Pipeline (`scripts/run_microgreen_demo.py`)

1. Load the best checkpoint (run33, 768px)
2. Preprocess each real photo (resize to 768x768, ImageNet normalize)
3. Run PyTorch inference to get per-pixel softmax probabilities
4. Apply iterative stem post-processing (see below)
5. Generate colored overlay visualizations

### Stem Post-Processing Algorithm

The raw model output (argmax) under-detected stems on real photos — thin, pale stems were often classified as background. An **iterative stem-growing algorithm** was developed to recover them:

**Step 1 — Identify stem candidates** using multiple cues:
- **HSV color filtering**: Hue 25–60, targeting green-yellow stem tissue
- **Canny edge proximity**: Candidates must be within 5px of a detected edge (structural features)
- **Green dominance**: G channel > R channel filters out brown soil
- **Not leaf**: Exclude pixels already classified as leaf

**Step 2 — Iterative dilation from leaf regions**:
- Start from detected leaf + model-predicted stem pixels as seed
- Use a **downward-biased asymmetric kernel** (11x21 px, anchor near top) for dilation — stems grow below leaves
- Each iteration, expand into neighboring stem candidates
- Repeat up to 60 iterations or until fewer than 5 new pixels are found

**Step 3 — Cleanup**:
- **Connected component analysis**: Only keep stem components that touch a (dilated) leaf region — removes isolated false positives
- **Morphological close**: Fill tiny gaps in stem segments
- **Width filter**: Erode away blobs wider than 20px — stems are thin structures

### Results

The microgreen demo produced overlay visualizations for approximately 39 real photos, showing the model successfully segments leaves and stems from real greenhouse imagery despite being trained exclusively on synthetic data. Results are stored in `exports/microgreen_results/`:

- Individual `*_overlay.png` files for each photo
- `microgreen_segmentation_demo.png` — 4-photo grid (original / mask / overlay)
- `microgreen_segmentation_slide.png` — 2-photo presentation-ready figure
- `stem_postprocess_comparison.png` — Side-by-side of model-only vs. post-processed stems
- `stem_threshold_comparison.png` — Effect of stem probability thresholds on pixel counts

---

## 7. Evaluation & Visualization Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_segmentation.py` | Main training loop with all loss functions, augmentation, and metrics |
| `scripts/confusion_matrix.py` | Generate pixel-level confusion matrix, recall, precision, and IoU charts |
| `scripts/validate_onnx.py` | Run ONNX model on test images, compute mIoU, save overlay visualizations |
| `scripts/gen_table.py` | Generate dark-themed experiment results table image (all 33 runs) |
| `scripts/compile_768_norm.py` | Compile 768px blob with baked-in normalization |
| `scripts/verify_norm.py` | Verify normalization is baked into blob via hash comparison |
| `scripts/run_microgreen_demo.py` | Run model on real microgreen photos with stem post-processing |
| `scripts/datasets/synthetic_plants.py` | Convert Supervisely/COCO dataset to canonical training format |
| `scripts/monitor_training.py` | Live training monitoring |
| `scripts/reshuffle_splits.py` | Re-generate train/val/test splits |

---

## 8. Summary

The GRAIL plant segmentation project achieved its goals through systematic experimentation:

- **33 training runs** exploring architecture, loss function, augmentation, resolution, and sampling strategies
- **Final model** (run33): LRASPP-MobileNetV3-Large, CE+Tversky loss, 768px, mIoU = **86.49%**, stem IoU = **79.6%**
- **Deployed** as a 7.8 MB `.blob` file on the OAK-D Pro's MyriadX VPU with normalization baked in for zero-preprocessing inference
- **Validated on real microgreen photos** with an iterative stem post-processing algorithm to compensate for domain gap on thin structures
- **Key insight**: Input resolution was the dominant factor (+15% mIoU from 256→768px), followed by loss function choice (CE+Tversky at high res)
