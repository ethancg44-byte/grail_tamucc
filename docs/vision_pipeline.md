# Machine Vision & Image Processing Pipeline

This document describes the complete vision pipeline running on the Raspberry Pi 5, from raw camera frame to robotic cut plan. It covers what the model does, what happens after the model produces its output, and how every processing stage works.

---

## Pipeline Overview

```
OAK-D Pro Camera
     |
     v
[1] Image Capture (RGB + Stereo Depth)
     |
     v
[2] On-Device Neural Network (MyriadX VPU)
     |  - Input: raw BGR 768x768
     |  - Output: 3-channel softmax → argmax → mono8 label mask
     |
     v
[3] Label Mask (0=BG, 1=LEAF, 2=STEM)
     |
     v
[4] Stem Skeletonization (Zhang-Suen thinning)
     |
     v
[5] Branch-Point Detection (3x3 neighbor counting)
     |
     v
[6] Temporal Node Tracking (nearest-neighbor association)
     |
     v
[7] 3D Projection (depth + camera intrinsics → XYZ)
     |
     v
[8] Rule-Based Cut Selection (Y-threshold + leaf density)
     |
     v
[9] RViz Visualization + Debug Overlay
```

---

## Stage 1 — Image Capture (`oak_camera_node.py`)

The OAK-D Pro camera provides three synchronized streams:

| Stream | Format | Description |
|--------|--------|-------------|
| RGB | BGR8, 640x480 @ 15fps | Color image from the center camera (CAM_A) |
| Depth | 16UC1 (uint16 mm) | Stereo depth map, aligned to RGB frame |
| CameraInfo | K matrix | Intrinsics (fx, fy, cx, cy) from factory calibration |

The DepthAI pipeline is built on-device:
- **StereoDepth** node fuses left + right mono cameras (400p) into a depth map using the `HIGH_DENSITY` preset with subpixel enabled
- **Depth alignment** to `CAM_A` (RGB) so every depth pixel corresponds to an RGB pixel
- **ImageManip** node resizes the RGB preview to the NN input size (768x768) and converts to `BGR888p` (planar) format
- **IR dot projector** can be enabled at 0.7 intensity for texture in featureless scenes

If the camera disconnects, the node automatically retries connection at a configurable interval (default 5s).

### ROS 2 Topics Published
- `/oak/rgb/image_raw` — BGR8 image
- `/oak/depth_aligned/image_raw` — 16UC1 depth in millimeters
- `/oak/camera_info` — Camera intrinsics
- `/oak/nn/segmentation_raw` — NN output label mask (mono8)

---

## Stage 2 — On-Device Semantic Segmentation

### The Model

**LRASPP-MobileNetV3-Large** — a lightweight semantic segmentation network trained on 8,000 synthetic plant images across 33 experimental runs.

- **Architecture**: MobileNetV3-Large backbone + LRASPP decoder
- **Classes**: Background (0), Leaf (1), Stem/Petiole (2)
- **Best run** (run33): mIoU = 86.49%, Stem IoU = 79.6%
- **Input**: 768x768 BGR uint8 (normalization baked into the blob)
- **Output**: (3, 768, 768) FP16 logits

### What "Normalization Baked In" Means

During training, images were normalized with ImageNet statistics:
```
mean = [0.485, 0.456, 0.406]  (RGB)
std  = [0.229, 0.224, 0.225]  (RGB)
```

Rather than performing this normalization on the Pi's CPU every frame, it was compiled directly into the blob during OpenVINO conversion:
```
--reverse_input_channels        # BGR → RGB
--mean_values=[123.675,116.28,103.53]   # mean * 255
--scale_values=[58.395,57.12,57.375]    # std * 255
```

This means the camera sends raw BGR uint8 pixels directly to the VPU with **zero preprocessing** on the Pi.

### Key Training Insights That Affect Deployment

| Finding | Implication |
|---------|-------------|
| Resolution is the #1 factor (+15% mIoU from 256→768px) | Deploy at 768px if possible; 640px as fallback |
| CE+Tversky loss penalizes missed stems | Model favors recall over precision for stems — slight over-prediction of stem boundaries is expected and acceptable |
| Stem IoU (79.6%) is lowest of all classes | Post-processing is needed to recover thin stems the model misses — see Stage 4+ |
| Trained on synthetic data only | Real-world performance depends on lighting/background similarity; the microgreen validation confirmed good transfer |

### NN Output Decoding (in `oak_camera_node.py`)

```python
# Get raw FP16 output from VPU
layer = in_nn.getFirstLayerFp16()
nn_out = np.array(layer, dtype=np.float32)

# Reshape from flat array to (3, H, W)
nn_out = nn_out.reshape(3, nn_h, nn_w)

# Argmax across class dimension → label mask
label_mask = np.argmax(nn_out, axis=0).astype(np.uint8)
# Result: 0=BG, 1=LEAF, 2=STEM for every pixel
```

This label mask is published as a `mono8` image and consumed by the perception node.

---

## Stage 3 — Label Mask Interpretation (`perception_node.py`)

The perception node receives the mono8 label mask and resizes it to match the RGB image dimensions (if they differ) using **nearest-neighbor interpolation** — critical because these are discrete class labels, not continuous values.

```python
label_mask = cv2.resize(nn_mask, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
```

The mask is then used in two parallel paths:
1. **Stem analysis** (Stages 4-7): Extract the stem class, skeletonize, find nodes, project to 3D
2. **Cut planning** (Stage 8): Combine node positions with leaf density to select pruning targets

---

## Stage 4 — Stem Skeletonization

### Purpose
The stem segmentation mask shows the full width of each stem (many pixels wide). For branch-point detection, we need to reduce each stem to a **1-pixel-wide skeleton** that preserves the topology (branching structure).

### Algorithm: Zhang-Suen Thinning

The system uses **Zhang-Suen thinning** — a classical morphological algorithm that iteratively erodes boundary pixels from a binary image while preserving connectivity:

1. Extract stem pixels: `stem_mask = (label_mask == 2) * 255`
2. Two sub-iterations per pass:
   - **Sub-iteration 1**: Removes south-east boundary pixels
   - **Sub-iteration 2**: Removes north-west boundary pixels
3. A pixel is removed only if:
   - It has 2-6 foreground neighbors (not isolated, not interior)
   - Exactly 1 zero-to-one transition in its 8-neighbor ring (preserves connectivity)
   - Specific directional conditions are met (prevents breaking the skeleton)
4. Repeat until no more pixels are removed

**Implementation**: Two backends are supported:
- **OpenCV ximgproc** (`cv2.ximgproc.thinning`) — fast C++ implementation, preferred
- **Pure NumPy fallback** — pixel-by-pixel Python, used when ximgproc is not available

The result is a binary image where every stem is exactly 1 pixel wide, maintaining all branches and junctions.

---

## Stage 5 — Branch-Point Detection

### Purpose
Branch points (where stems fork) are the most informative locations for understanding plant structure and selecting pruning sites.

### Algorithm

1. **Neighbor counting**: Convolve the skeleton with a 3x3 kernel (center=0, all neighbors=1):
   ```
   [1 1 1]
   [1 0 1]
   [1 1 1]
   ```
   This counts how many skeleton neighbors each pixel has.

2. **Branch-point criterion**: A skeleton pixel with **3 or more neighbors** is a branch point (where two or more skeleton segments meet).

3. **Clustering**: Nearby branch points (within `2 * branch_kernel_size` pixels) are merged into a single point at their centroid. This prevents a single junction from producing multiple detections.

### Output
A list of `(u, v)` pixel coordinates — each one a detected branch point in the image.

---

## Stage 6 — Temporal Node Tracking

### Purpose
Single-frame branch-point detection is noisy — points appear and disappear between frames due to slight mask changes. Temporal tracking stabilizes the detections across frames for reliable 3D projection.

### Algorithm: Nearest-Neighbor Association with Age

Each tracked node has: `{id, u, v, age}`

Per frame:
1. **Age all existing nodes**: `age += 1`
2. **Associate detections**: For each tracked node, find the nearest new detection within `node_association_max_dist_px` (default 30px). If found, update the node's position and reset age to 0.
3. **Create new nodes**: Any unmatched detection becomes a new tracked node with a unique ID.
4. **Prune stale nodes**: Remove nodes with `age > node_max_age_frames` (default 10 frames) — they haven't been seen recently enough to trust.

### Confidence
Node confidence decays linearly with age:
```python
confidence = max(0.0, 1.0 - age / node_max_age)
```
A freshly detected node has confidence 1.0; a node not seen for 5 frames (with max_age=10) has confidence 0.5.

---

## Stage 7 — 3D Projection

### Purpose
Convert 2D pixel coordinates (u, v) to 3D camera-frame coordinates (X, Y, Z) using the depth map and camera intrinsics. This is necessary for the robot arm to know where to reach.

### Algorithm: Pinhole Camera Back-Projection

Given pixel (u, v), depth Z (from the depth map), and intrinsics (fx, fy, cx, cy):

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth[v, u] / 1000.0   (convert mm → meters)
```

### Depth Validation
- **Bounds check**: `u` and `v` must be within the depth image dimensions
- **Range filter**: Depth must be between `depth_min_mm` (100mm) and `depth_max_mm` (3000mm)
- Invalid depths (0, out-of-range, or out-of-bounds) produce a zero-valued `PointStamped` — downstream stages handle this gracefully

### Output
Each stem node gets a `PointStamped` message with XYZ coordinates in the `oak_camera_frame`.

---

## Stage 8 — Rule-Based Cut Selection

### Purpose
Select which stem nodes are candidates for robotic pruning based on plant structure heuristics.

### Rules

**Rule 1 — Near Crown (Y-threshold)**:
If a node's `v` coordinate exceeds `cut_y_threshold_px` (default 300), it is near the base/crown of the plant. These are prioritized because cutting near the crown promotes regrowth.
- Adds reason: `NEAR_CROWN`
- Confidence factor: 0.6

**Rule 2 — Overcrowding (Leaf Density)**:
Measure the fraction of leaf pixels in a square region (default radius 50px) around the node:
```python
density = leaf_pixels_in_region / total_pixels_in_region
```
If density exceeds `cut_leaf_density_threshold` (default 0.4), the area is overcrowded.
- Adds reason: `OVERCROWDING`
- Confidence factor: the density value itself

**Confidence Scoring**:
- Final confidence = mean of all triggered rule factors
- Only targets with confidence >= `cut_min_confidence` (0.3) are published
- Reason codes are comma-joined (e.g., `NEAR_CROWN,OVERCROWDING`)

### Output
A `CutPlan` message containing a list of `CutTarget` messages, each with:
- 3D point in camera frame
- Confidence score
- Reason code string

---

## Stage 9 — Visualization

### RViz Markers
- **Green spheres**: Stem nodes (transparency = confidence)
- **Red spheres**: Cut targets (larger, opaque)
- Lifetime: 1 second (auto-cleanup if not refreshed)

### Debug Overlay (`/debug/overlay_image`)
A composite image showing all pipeline outputs on top of the RGB feed:
- Semi-transparent class colors (green=leaf, orange=stem, black=BG) at 35% opacity
- **Yellow skeleton** lines drawn where the thinned stem is
- **Green circles** at branch points (with white border)
- **Red X marks** at cut targets with reason code text

This overlay is invaluable for debugging — it shows exactly what the model sees, where the skeleton is, which nodes were detected, and which were selected for cutting.

---

## Configuration Parameters

All parameters are set in `config/perception_params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skeleton_method` | `opencv` | `opencv` (ximgproc) or `zhang_suen` (fallback) |
| `branch_kernel_size` | 3 | Branch-point clustering radius |
| `node_association_max_dist_px` | 30 | Max pixel distance to associate a detection with a tracked node |
| `node_max_age_frames` | 10 | Frames before an unseen node is pruned |
| `depth_min_mm` | 100 | Minimum valid depth (mm) |
| `depth_max_mm` | 3000 | Maximum valid depth (mm) |
| `cut_y_threshold_px` | 300 | Y pixel threshold for NEAR_CROWN rule |
| `cut_leaf_density_radius_px` | 50 | Region radius for leaf density measurement |
| `cut_leaf_density_threshold` | 0.4 | Density above which OVERCROWDING is triggered |
| `cut_min_confidence` | 0.3 | Minimum confidence to publish a cut target |
| `publish_debug_overlay` | true | Whether to publish the debug overlay image |

Camera parameters in `config/oak_params.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resolution_width` | 640 | RGB stream width |
| `resolution_height` | 480 | RGB stream height |
| `fps` | 15 | Frame rate |
| `model_blob_path` | `''` | Path to the .blob file |
| `enable_mock_nn` | true | Use synthetic data (no hardware needed) |
| `nn_input_width` | 768 | NN input width (must match blob) |
| `nn_input_height` | 768 | NN input height (must match blob) |

---

## Available Model Blobs

| File | Resolution | Notes |
|------|-----------|-------|
| `exports/plant_seg_lraspp.blob` | 768x768 | Best accuracy (mIoU=0.865), use if FPS is acceptable |
| `exports/plant_seg_lraspp_640.blob` | 640x640 | Faster inference, slightly lower accuracy (mIoU=0.846) |

Both blobs have ImageNet normalization and BGR→RGB conversion baked in. Pass raw camera frames directly — no CPU preprocessing needed.

When switching blobs, update **both** `model_blob_path` and `nn_input_width`/`nn_input_height` in `oak_params.yaml`.

---

## Microgreen Validation (Stem Post-Processing Reference)

During development, we validated the model on real microgreen photographs and found that raw argmax under-detects stems. An iterative stem-growing algorithm was developed as a reference for potential runtime enhancement:

1. **Color filtering** (HSV hue 25-60) to identify stem-colored tissue
2. **Canny edge proximity** (within 5px) to ensure structural features
3. **Green dominance** (G > R) to exclude soil
4. **Downward-biased dilation** from leaf regions using an asymmetric kernel (3:1 downward bias)
5. **Connected component filtering** — only keep stems touching leaf regions
6. **Width filtering** — erode blobs wider than 20px

This post-processing is implemented in `scripts/run_microgreen_demo.py` and could be adapted into `perception_node.py` if runtime stem detection needs improvement. Results are in `exports/microgreen_results/`.

---

## Summary of Post-Model Processing

After the neural network produces its per-pixel class labels, the following computer vision techniques are applied:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| Skeletonization | Zhang-Suen morphological thinning | Reduce stem mask to 1px-wide topology |
| Branch detection | 3x3 convolution + neighbor counting | Find junction points where stems fork |
| Temporal tracking | Nearest-neighbor association + age decay | Stabilize noisy per-frame detections |
| 3D projection | Pinhole back-projection with depth | Convert pixel coords to robot-reachable 3D points |
| Cut planning | Y-threshold + leaf density heuristics | Select which nodes to prune |
| Visualization | Overlay compositing | Debug feedback showing all pipeline stages |

Each stage builds on the previous one, transforming raw pixel labels into actionable 3D pruning targets for the robotic arm.
