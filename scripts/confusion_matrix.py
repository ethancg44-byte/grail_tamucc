#!/usr/bin/env python3
"""Generate a pixel-level confusion matrix for a trained segmentation model.

Usage:
  python scripts/confusion_matrix.py \
    --checkpoint checkpoints/run30_cedice_stemsamp_512/best_model.pth \
    --data_dir data/synthetic_plants \
    --split val
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, deeplabv3_mobilenet_v3_large
from tqdm import tqdm


NUM_CLASSES = 3
CLASS_NAMES = ['BACKGROUND', 'LEAF', 'STEM OR PETIOLE']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model(num_classes, arch='lraspp'):
    if arch == 'deeplabv3':
        model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
    else:
        model = lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
    return model


def load_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get('arch', 'lraspp')
    input_size = ckpt.get('input_size', 256)
    num_classes = ckpt.get('num_classes', NUM_CLASSES)

    model = build_model(num_classes, arch=arch)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    epoch = ckpt.get('epoch', '?')
    best_miou = ckpt.get('best_miou', '?')
    print(f'Loaded {arch} from {checkpoint_path}')
    print(f'  Epoch: {epoch}, Best mIoU: {best_miou}, Input size: {input_size}')
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


def compute_confusion_matrix(model, data_dir, split, input_size, device):
    data_dir = Path(data_dir)
    split_file = data_dir / 'splits' / f'{split}.txt'
    sample_names = [l.strip() for l in split_file.read_text().strip().split('\n') if l.strip()]

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for name in tqdm(sample_names, desc=f'Evaluating {split}'):
        img_path = data_dir / 'images' / f'{name}.png'
        mask_path = data_dir / 'masks' / f'{name}.png'

        image_bgr = cv2.imread(str(img_path))
        mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or mask_gt is None:
            print(f'  Warning: skipping {name} (failed to load)')
            continue

        pred = predict(model, image_bgr, input_size, device)

        # Resize GT mask to match prediction size
        mask_gt = cv2.resize(mask_gt, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

        # Clip to valid class range
        mask_gt = np.clip(mask_gt, 0, NUM_CLASSES - 1)

        # Accumulate confusion matrix
        for gt_class in range(NUM_CLASSES):
            for pred_class in range(NUM_CLASSES):
                cm[gt_class, pred_class] += np.sum(
                    (mask_gt == gt_class) & (pred == pred_class)
                )

    return cm


def print_confusion_matrix(cm):
    print('\n' + '=' * 70)
    print('CONFUSION MATRIX (rows=GT, cols=Predicted)')
    print('=' * 70)

    # Raw counts
    header = f'{"":>18s} | ' + ' | '.join(f'{name:>12s}' for name in CLASS_NAMES) + ' | Total'
    print(header)
    print('-' * len(header))
    for i, name in enumerate(CLASS_NAMES):
        row_total = cm[i].sum()
        row = f'{name:>18s} | ' + ' | '.join(f'{cm[i, j]:>12,d}' for j in range(NUM_CLASSES))
        row += f' | {row_total:>12,d}'
        print(row)

    col_totals = cm.sum(axis=0)
    total_row = f'{"Total":>18s} | ' + ' | '.join(f'{col_totals[j]:>12,d}' for j in range(NUM_CLASSES))
    total_row += f' | {cm.sum():>12,d}'
    print('-' * len(header))
    print(total_row)

    # Row-normalized (recall: what % of each GT class was predicted as X)
    print('\n' + '=' * 70)
    print('ROW-NORMALIZED (Recall: how each GT class is predicted)')
    print('=' * 70)
    header_pct = f'{"":>18s} | ' + ' | '.join(f'{name:>12s}' for name in CLASS_NAMES) + ' | Recall'
    print(header_pct)
    print('-' * len(header_pct))
    for i, name in enumerate(CLASS_NAMES):
        row_sum = cm[i].sum()
        if row_sum == 0:
            pcts = [0.0] * NUM_CLASSES
            recall = 0.0
        else:
            pcts = [cm[i, j] / row_sum * 100 for j in range(NUM_CLASSES)]
            recall = cm[i, i] / row_sum * 100
        row = f'{name:>18s} | ' + ' | '.join(f'{p:>11.2f}%' for p in pcts)
        row += f' | {recall:>10.2f}%'
        print(row)

    # Column-normalized (precision: of everything predicted as X, what % was correct)
    print('\n' + '=' * 70)
    print('COL-NORMALIZED (Precision: accuracy of each prediction)')
    print('=' * 70)
    header_pct2 = f'{"":>18s} | ' + ' | '.join(f'{name:>12s}' for name in CLASS_NAMES) + ' | '
    print(header_pct2)
    print('-' * len(header_pct2))
    for i, name in enumerate(CLASS_NAMES):
        pcts = []
        for j in range(NUM_CLASSES):
            col_sum = cm[:, j].sum()
            pcts.append(cm[i, j] / col_sum * 100 if col_sum > 0 else 0.0)
        row = f'{name:>18s} | ' + ' | '.join(f'{p:>11.2f}%' for p in pcts)
        print(row)

    prec_row = f'{"Precision":>18s} | '
    for j in range(NUM_CLASSES):
        col_sum = cm[:, j].sum()
        prec = cm[j, j] / col_sum * 100 if col_sum > 0 else 0.0
        prec_row += f'{prec:>11.2f}% | '
    print('-' * len(header_pct2))
    print(prec_row)

    # Per-class IoU
    print('\n' + '=' * 70)
    print('PER-CLASS IoU')
    print('=' * 70)
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        print(f'  {name:>18s}: {iou:.4f} ({iou*100:.2f}%)')

    overall_acc = np.diag(cm).sum() / cm.sum() * 100
    miou = np.mean([cm[i, i] / (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])
                     for i in range(NUM_CLASSES)])
    print(f'\n  Overall pixel accuracy: {overall_acc:.2f}%')
    print(f'  Mean IoU: {miou:.4f} ({miou*100:.2f}%)')


def plot_confusion_matrix(cm, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Raw counts (log scale for visibility)
    ax1 = axes[0]
    cm_log = np.log10(cm.astype(float) + 1)
    im1 = ax1.imshow(cm_log, cmap='Blues')
    ax1.set_xticks(range(NUM_CLASSES))
    ax1.set_yticks(range(NUM_CLASSES))
    ax1.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Ground Truth')
    ax1.set_title('Raw Counts (log₁₀)')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax1.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=8,
                     color='white' if cm_log[i, j] > cm_log.max() * 0.85 else 'black')

    # Row-normalized (recall)
    ax2 = axes[1]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_recall = np.where(row_sums > 0, cm / row_sums * 100, 0)
    im2 = ax2.imshow(cm_recall, cmap='Blues', vmin=0, vmax=100)
    ax2.set_xticks(range(NUM_CLASSES))
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Ground Truth')
    ax2.set_title('Recall')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax2.text(j, i, f'{cm_recall[i, j]:.1f}%', ha='center', va='center', fontsize=10,
                     color='white' if cm_recall[i, j] > 50 else 'black')
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Column-normalized (precision)
    ax3 = axes[2]
    col_sums = cm.sum(axis=0, keepdims=True)
    cm_prec = np.where(col_sums > 0, cm / col_sums * 100, 0)
    im3 = ax3.imshow(cm_prec, cmap='Blues', vmin=0, vmax=100)
    ax3.set_xticks(range(NUM_CLASSES))
    ax3.set_yticks(range(NUM_CLASSES))
    ax3.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Ground Truth')
    ax3.set_title('Precision')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax3.text(j, i, f'{cm_prec[i, j]:.1f}%', ha='center', va='center', fontsize=10,
                     color='white' if cm_prec[i, j] > 50 else 'black')
    plt.colorbar(im3, ax=ax3, shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved confusion matrix plot to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Compute confusion matrix for segmentation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Canonical dataset directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate (default: val)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot (default: <checkpoint_dir>/confusion_matrix.png)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model, input_size = load_checkpoint(args.checkpoint, device)
    cm = compute_confusion_matrix(model, args.data_dir, args.split, input_size, device)

    print_confusion_matrix(cm)

    output_path = args.output or str(Path(args.checkpoint).parent / f'confusion_matrix_{args.split}.png')
    plot_confusion_matrix(cm, output_path)


if __name__ == '__main__':
    main()
