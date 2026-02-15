# Model Training & Export Pipeline

Step-by-step guide: dataset preparation, training, ONNX export, OpenVINO conversion, and DepthAI blob compilation for the OAK-D Pro.

## Prerequisites

**Training workstation** (not the Pi 5 — needs a GPU):
```bash
cd ~/ros2_ws
pip install -r requirements-training.txt
```

**Pi 5** (deployment only):
```bash
pip install depthai  # already installed from Phase 1
```

## 1. Dataset Preparation

### Option A: Synthetic Plants (recommended starter)

Download from [Dataset Ninja](https://datasetninja.com/synthetic-plants) in Supervisely or COCO format, then convert:

```bash
# Supervisely format
python scripts/datasets/synthetic_plants.py \
  --raw_dir data/synthetic_plants_raw \
  --output_dir data/synthetic_plants

# COCO format
python scripts/datasets/synthetic_plants.py \
  --raw_dir data/synthetic_plants_coco \
  --output_dir data/synthetic_plants
```

### Option B: CVPPP LSC

Download from [CVPPP challenge](https://www.plant-phenotyping.org/datasets-home), then convert:

```bash
python scripts/datasets/cvppp.py \
  --raw_dir data/cvppp_raw \
  --output_dir data/cvppp
```

> **Note:** CVPPP only contains leaf labels (no stem/petiole). Combine with Synthetic Plants for full 3-class training.

### Canonical Format

After conversion, each dataset directory contains:
```
data/<dataset_name>/
  images/*.png      # RGB images (256x256)
  masks/*.png       # Label masks (mono8: 0=BG, 1=LEAF, 2=STEM)
  labels.yaml       # Class mapping metadata
  splits/
    train.txt       # 80% — one filename per line (no extension)
    val.txt         # 10%
    test.txt        # 10%
```

## 2. Training

### Basic training (single dataset)

```bash
python scripts/train_segmentation.py \
  --data_dirs data/synthetic_plants \
  --output_dir checkpoints/run01 \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.001
```

### Multi-dataset training

```bash
python scripts/train_segmentation.py \
  --data_dirs data/synthetic_plants data/cvppp \
  --output_dir checkpoints/run02 \
  --epochs 50 \
  --batch_size 8
```

### Resume from checkpoint

```bash
python scripts/train_segmentation.py \
  --data_dirs data/synthetic_plants \
  --output_dir checkpoints/run01 \
  --resume checkpoints/run01/latest_checkpoint.pth \
  --epochs 100
```

### Train + auto-export ONNX

```bash
python scripts/train_segmentation.py \
  --data_dirs data/synthetic_plants \
  --output_dir checkpoints/run01 \
  --epochs 50 \
  --export_onnx
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dirs` | (required) | One or more dataset directories |
| `--output_dir` | `checkpoints/default` | Checkpoint output directory |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 0.001 | Learning rate (AdamW) |
| `--input_size` | 256 | Input resolution |
| `--class_weights` | 0.5 2.0 3.0 | Loss weights: BG, LEAF, STEM |
| `--resume` | — | Path to checkpoint to resume |
| `--export_onnx` | false | Export ONNX after training |
| `--no_pretrained` | false | Skip ImageNet pretrained weights |

### Training outputs

```
checkpoints/run01/
  train_config.yaml       # Saved configuration
  best_model.pth          # Best model by mIoU
  latest_checkpoint.pth   # Latest checkpoint (every 10 epochs)
  training_curves.png     # Loss and mIoU plots
  model.onnx              # ONNX export (if --export_onnx)
```

## 3. ONNX Validation

Verify the exported ONNX model before proceeding:

```bash
python scripts/validate_onnx.py \
  --onnx_path checkpoints/run01/model.onnx \
  --image_dir data/synthetic_plants/images \
  --mask_dir data/synthetic_plants/masks \
  --output_dir exports/validation_results \
  --max_images 100
```

Check the visualizations in `exports/validation_results/` and the reported mIoU.

## 4. OpenVINO Conversion

Convert ONNX to OpenVINO IR with normalization baked in:

```bash
./scripts/export_to_openvino.sh exports/plant_seg_lraspp.onnx exports/openvino
```

This runs:
```bash
mo \
  --input_model exports/plant_seg_lraspp.onnx \
  --output_dir exports/openvino \
  --compress_to_fp16 \
  --reverse_input_channels \
  --mean_values "[123.675,116.28,103.53]" \
  --scale_values "[58.395,57.12,57.375]" \
  --input_shape "[1,3,256,256]"
```

Key flags:
- `--reverse_input_channels`: OAK sends BGR, model expects RGB — bakes the swap into IR
- `--mean_values` / `--scale_values`: Bakes ImageNet normalization — OAK can send raw uint8
- `--compress_to_fp16`: Required for Myriad X (OAK-D Pro VPU)

## 5. Blob Compilation

Compile OpenVINO IR to DepthAI `.blob`:

```bash
./scripts/compile_blob.sh exports/openvino/plant_seg_lraspp.xml exports 6
```

This uses `blobconverter` (Luxonis cloud service). The number `6` is the number of VPU shaves (6 is a good default for OAK-D Pro).

## 6. Deployment to Pi 5

1. Copy the blob to the Pi:
   ```bash
   scp exports/plant_seg_lraspp.blob pi@<pi-ip>:~/ros2_ws/exports/
   ```

2. Update `src/bringup/config/oak_params.yaml`:
   ```yaml
   oak_camera_node:
     ros__parameters:
       model_blob_path: '/home/<user>/ros2_ws/exports/plant_seg_lraspp.blob'
       enable_mock_nn: false
   ```

3. Rebuild and launch:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select bringup --symlink-install
   source install/setup.bash
   ros2 launch bringup bringup_perception.launch.py
   ```

## Integration Contract

The OAK camera node (`oak_camera_node.py`) expects:
- **Input**: BGR888p (planar BGR uint8) resized to 256x256 by ImageManip
- **Output**: FP16 flat array reshaped to `(3, 256, 256)` → argmax → mono8 label mask
- Labels: 0=BACKGROUND, 1=LEAF, 2=STEM_OR_PETIOLE

The export pipeline ensures this contract is met:
1. Training uses RGB + ImageNet normalization
2. OpenVINO `mo` bakes `--reverse_input_channels` (BGR→RGB) and normalization into the IR
3. The blob receives raw BGR uint8 from OAK and handles everything internally

## Troubleshooting

**ONNX export fails**: Ensure `opset_version=12` (set in training script). Some PyTorch ops need specific opset versions.

**OpenVINO mo errors**: Check `openvino-dev` version matches your ONNX opset. Try `pip install --upgrade openvino-dev`.

**blobconverter fails**: The cloud service may be down. Use local `compile_tool` as described in `compile_blob.sh` comments.

**Poor inference quality on OAK**: Verify normalization values match exactly. Compare ONNX Runtime output vs OAK output on the same image.
