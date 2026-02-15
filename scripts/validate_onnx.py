#!/usr/bin/env python3
"""Validate an ONNX segmentation model.

Runs inference on a folder of test images using ONNX Runtime, optionally
computes mIoU against ground-truth masks, and saves color-coded visualizations.

Usage:
  # Basic inference + visualization
  python validate_onnx.py \\
    --onnx_path exports/plant_seg_lraspp.onnx \\
    --image_dir data/synthetic_plants/images \\
    --output_dir exports/validation_results

  # With ground-truth masks for mIoU
  python validate_onnx.py \\
    --onnx_path exports/plant_seg_lraspp.onnx \\
    --image_dir data/synthetic_plants/images \\
    --mask_dir data/synthetic_plants/masks \\
    --output_dir exports/validation_results

  # Limit number of images
  python validate_onnx.py \\
    --onnx_path exports/plant_seg_lraspp.onnx \\
    --image_dir data/synthetic_plants/images \\
    --output_dir exports/validation_results \\
    --max_images 50
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
CLASS_NAMES = ['BACKGROUND', 'LEAF', 'STEM_OR_PETIOLE']

# ImageNet normalization (must match training preprocessing)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Color map for visualization (BGR for cv2)
COLOR_MAP = {
    0: (0, 0, 0),       # BACKGROUND — black
    1: (0, 200, 0),     # LEAF — green
    2: (0, 100, 255),   # STEM_OR_PETIOLE — orange
}


def preprocess(img_bgr: np.ndarray, input_size: int) -> np.ndarray:
    """Preprocess image for ONNX inference.

    Converts BGR to RGB, resizes, normalizes with ImageNet stats,
    and converts to NCHW float32 tensor.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size),
                             interpolation=cv2.INTER_LINEAR)

    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → NCHW
    img_nchw = img_norm.transpose(2, 0, 1)[np.newaxis, ...]
    return img_nchw


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert mono8 label mask to color BGR visualization."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, bgr in COLOR_MAP.items():
        color[mask == label_id] = bgr
    return color


def overlay(img_bgr: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay colored mask on image."""
    img_resized = cv2.resize(img_bgr, (mask_color.shape[1], mask_color.shape[0]))
    return cv2.addWeighted(img_resized, 1 - alpha, mask_color, alpha, 0)


def compute_iou(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute per-class IoU and mIoU."""
    ious = {}
    for cls in range(NUM_CLASSES):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            ious[CLASS_NAMES[cls]] = float('nan')
        else:
            ious[CLASS_NAMES[cls]] = float(intersection) / float(union)

    valid = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = float(np.mean(valid)) if valid else 0.0
    return ious


def main():
    parser = argparse.ArgumentParser(
        description='Validate ONNX segmentation model on test images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--onnx_path', type=str, required=True,
        help='Path to ONNX model file',
    )
    parser.add_argument(
        '--image_dir', type=str, required=True,
        help='Directory containing test images (PNG/JPG)',
    )
    parser.add_argument(
        '--mask_dir', type=str, default=None,
        help='Directory containing ground-truth masks (for mIoU computation)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='exports/validation_results',
        help='Directory for output visualizations (default: exports/validation_results)',
    )
    parser.add_argument(
        '--input_size', type=int, default=256,
        help='Model input size (default: 256)',
    )
    parser.add_argument(
        '--max_images', type=int, default=0,
        help='Max images to process (0 = all, default: 0)',
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ONNX model
    print(f'Loading ONNX model: {args.onnx_path}')
    session = ort.InferenceSession(args.onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape
    print(f'  Input:  {input_name} {input_shape}')
    print(f'  Output: {output_name} {output_shape}')

    # Find images
    image_dir = Path(args.image_dir)
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    )
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f'Processing {len(image_paths)} images...')

    # Run inference
    all_ious = []
    for i, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f'  WARNING: Could not read {img_path.name}, skipping')
            continue

        # Preprocess and run inference
        input_tensor = preprocess(img_bgr, args.input_size)
        outputs = session.run([output_name], {input_name: input_tensor})
        logits = outputs[0]  # Shape: (1, num_classes, H, W)

        # Argmax to get label mask
        pred_mask = np.argmax(logits[0], axis=0).astype(np.uint8)

        # Compute IoU if GT mask available
        if args.mask_dir:
            mask_path = Path(args.mask_dir) / img_path.name
            if not mask_path.exists():
                mask_path = Path(args.mask_dir) / (img_path.stem + '.png')
            if mask_path.exists():
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    gt_resized = cv2.resize(
                        gt_mask, (args.input_size, args.input_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    ious = compute_iou(pred_mask, gt_resized)
                    all_ious.append(ious)

        # Save visualization
        mask_color = colorize_mask(pred_mask)
        overlay_img = overlay(img_bgr, mask_color)

        vis_name = f'{img_path.stem}_pred.png'
        cv2.imwrite(str(output_dir / vis_name), overlay_img)

        if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
            print(f'  Processed {i + 1}/{len(image_paths)}')

    # Report aggregate metrics
    if all_ious:
        print(f'\n--- Metrics over {len(all_ious)} images ---')
        for cls_name in CLASS_NAMES + ['mIoU']:
            values = [d[cls_name] for d in all_ious if not np.isnan(d.get(cls_name, float('nan')))]
            if values:
                mean_val = np.mean(values)
                print(f'  {cls_name:20s}: {mean_val:.4f}')
            else:
                print(f'  {cls_name:20s}: N/A')
    else:
        print('\nNo ground-truth masks provided — skipping mIoU computation.')

    print(f'\nVisualizations saved to {output_dir}')


if __name__ == '__main__':
    main()
