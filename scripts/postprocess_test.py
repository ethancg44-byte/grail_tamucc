#!/usr/bin/env python3
"""Test morphological post-processing on stem predictions.

Evaluates different erosion kernel sizes on the stem class to find the
optimal post-processing that improves stem precision/IoU by trimming
over-extended boundary predictions.

Usage:
  python scripts/postprocess_test.py \
    --checkpoint checkpoints/run31_cetversky_stemsamp_512/best_model.pth \
    --data_dir data/synthetic_plants \
    --split val
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

NUM_CLASSES = 3
CLASS_NAMES = ['BACKGROUND', 'LEAF', 'STEM OR PETIOLE']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Stem class index
STEM_CLASS = 2


def load_checkpoint(checkpoint_path, device):
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get('arch', 'lraspp')
    input_size = ckpt.get('input_size', 256)
    num_classes = ckpt.get('num_classes', NUM_CLASSES)
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'Loaded {arch} from {checkpoint_path}')
    print(f'  Epoch: {ckpt.get("epoch", "?")}, Best mIoU: {ckpt.get("best_miou", "?")}')
    return model, input_size


def predict(model, image_bgr, input_size, device):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)['out']
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def apply_stem_erosion(pred, kernel_size):
    """Erode the stem class prediction to trim boundary false positives."""
    if kernel_size <= 0:
        return pred
    stem_mask = (pred == STEM_CLASS).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(stem_mask, kernel, iterations=1)
    # Pixels that were stem but got eroded -> assign to background
    result = pred.copy()
    result[(stem_mask == 1) & (eroded == 0)] = 0  # background
    return result


def apply_stem_opening(pred, kernel_size):
    """Opening (erosion + dilation) on stem class to remove small FP regions."""
    if kernel_size <= 0:
        return pred
    stem_mask = (pred == STEM_CLASS).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(stem_mask, cv2.MORPH_OPEN, kernel)
    result = pred.copy()
    result[(stem_mask == 1) & (opened == 0)] = 0
    return result


def compute_metrics(cm):
    """Compute per-class precision, recall, IoU from confusion matrix."""
    metrics = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[name] = {'precision': precision, 'recall': recall, 'iou': iou}
    miou = np.mean([m['iou'] for m in metrics.values()])
    metrics['mIoU'] = miou
    return metrics


def evaluate_postprocess(model, data_dir, split, input_size, device, postprocess_fn, kernel_size):
    data_dir = Path(data_dir)
    split_file = data_dir / 'splits' / f'{split}.txt'
    sample_names = [l.strip() for l in split_file.read_text().strip().split('\n') if l.strip()]

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for name in tqdm(sample_names, desc=f'k={kernel_size}', leave=False):
        img_path = data_dir / 'images' / f'{name}.png'
        mask_path = data_dir / 'masks' / f'{name}.png'

        image_bgr = cv2.imread(str(img_path))
        mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask_gt is None:
            continue

        pred = predict(model, image_bgr, input_size, device)
        pred = postprocess_fn(pred, kernel_size)

        mask_gt = cv2.resize(mask_gt, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
        mask_gt = np.clip(mask_gt, 0, NUM_CLASSES - 1)

        for gt_class in range(NUM_CLASSES):
            for pred_class in range(NUM_CLASSES):
                cm[gt_class, pred_class] += np.sum(
                    (mask_gt == gt_class) & (pred == pred_class)
                )

    return cm


def main():
    parser = argparse.ArgumentParser(description='Test post-processing on stem predictions')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model, input_size = load_checkpoint(args.checkpoint, device)

    # Test configurations: (name, function, kernel_sizes)
    configs = [
        ('No post-processing', lambda p, k: p, [0]),
        ('Erosion', apply_stem_erosion, [3, 5, 7]),
        ('Opening', apply_stem_opening, [3, 5, 7, 9]),
    ]

    results = []

    for config_name, fn, kernel_sizes in configs:
        for k in kernel_sizes:
            label = f'{config_name}' if k == 0 else f'{config_name} (k={k})'
            print(f'\nEvaluating: {label}...')
            cm = evaluate_postprocess(model, args.data_dir, args.split, input_size, device, fn, k)
            metrics = compute_metrics(cm)
            results.append((label, metrics))

            stem = metrics['STEM OR PETIOLE']
            print(f'  Stem  -> Prec: {stem["precision"]*100:.2f}%  Recall: {stem["recall"]*100:.2f}%  IoU: {stem["iou"]*100:.2f}%')
            print(f'  mIoU: {metrics["mIoU"]*100:.2f}%')

    # Summary table
    print('\n' + '=' * 90)
    print('SUMMARY: Post-processing impact on stem metrics')
    print('=' * 90)
    header = f'{"Method":<25s} | {"Stem Prec":>10s} | {"Stem Recall":>12s} | {"Stem IoU":>10s} | {"mIoU":>10s}'
    print(header)
    print('-' * len(header))

    baseline_iou = results[0][1]['STEM OR PETIOLE']['iou']
    baseline_miou = results[0][1]['mIoU']

    for label, metrics in results:
        stem = metrics['STEM OR PETIOLE']
        iou_delta = stem['iou'] - baseline_iou
        miou_delta = metrics['mIoU'] - baseline_miou
        row = f'{label:<25s} | {stem["precision"]*100:>9.2f}% | {stem["recall"]*100:>11.2f}% | {stem["iou"]*100:>9.2f}% | {metrics["mIoU"]*100:>9.2f}%'
        if iou_delta != 0:
            row += f'  (stem {iou_delta*100:+.2f}%, mIoU {miou_delta*100:+.2f}%)'
        print(row)

    print('=' * 90)


if __name__ == '__main__':
    main()
