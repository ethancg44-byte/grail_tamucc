#!/usr/bin/env python3
"""Train LRASPP-MobileNetV3-Large for plant segmentation.

Trains a 3-class semantic segmentation model (BG, LEAF, STEM_OR_PETIOLE)
using the canonical dataset format produced by the dataset adapters.

Usage:
  python train_segmentation.py \\
    --data_dirs data/synthetic_plants data/cvppp \\
    --output_dir checkpoints/run01 \\
    --epochs 50 --batch_size 8 --lr 0.001

  # Resume from checkpoint
  python train_segmentation.py \\
    --data_dirs data/synthetic_plants \\
    --output_dir checkpoints/run01 \\
    --resume checkpoints/run01/best_model.pth

  # Train and export ONNX
  python train_segmentation.py \\
    --data_dirs data/synthetic_plants \\
    --output_dir checkpoints/run01 \\
    --export_onnx
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
CLASS_NAMES = ['BACKGROUND', 'LEAF', 'STEM_OR_PETIOLE']

# ImageNet normalization (applied in training; baked into OpenVINO IR at export)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlantSegDataset(Dataset):
    """Reads a canonical dataset directory (images/, masks/, splits/)."""

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        input_size: int = 256,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augment = augment

        split_file = self.data_dir / 'splits' / f'{split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f'Split file not found: {split_file}')

        self.sample_names = [
            line.strip() for line in split_file.read_text().strip().split('\n')
            if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        name = self.sample_names[idx]
        img_path = self.data_dir / 'images' / f'{name}.png'
        mask_path = self.data_dir / 'masks' / f'{name}.png'

        # Load image as RGB (cv2 reads BGR)
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f'Image not found: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Mask not found: {mask_path}')

        # Ensure correct size
        if img.shape[0] != self.input_size or img.shape[1] != self.input_size:
            img = cv2.resize(img, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.input_size, self.input_size),
                              interpolation=cv2.INTER_NEAREST)

        # Augmentations (applied jointly to image and mask)
        if self.augment:
            img, mask = self._augment(img, mask)

        # Normalize image
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        # To tensor: HWC → CHW
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64)).long()

        return img_tensor, mask_tensor

    @staticmethod
    def _augment(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply joint augmentations to image and mask."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        # Random brightness/contrast (image only)
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-20, 20)     # brightness
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Random color jitter (image only)
        if np.random.random() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img, mask


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class SegmentationModelWrapper(nn.Module):
    """Wraps torchvision segmentation model to output plain tensor.

    torchvision LRASPP returns a dict {'out': tensor}. This wrapper
    extracts the 'out' tensor so ONNX export gets a clean single-output graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            return out['out']
        return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute per-class IoU and mean IoU."""
    ious = {}
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            ious[CLASS_NAMES[cls]] = float('nan')
        else:
            ious[CLASS_NAMES[cls]] = float(intersection) / float(union)

    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = float(np.mean(valid_ious)) if valid_ious else 0.0
    return ious


def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t in range(num_classes):
        for p in range(num_classes):
            cm[t, p] = np.sum((target == t) & (pred == p))
    return cm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} [train]', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Handle dict output (when not using wrapper during training)
        if isinstance(outputs, dict):
            outputs = outputs['out']

        # Resize outputs if needed (LRASPP may output slightly different size)
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Validate model, return loss, IoU dict, and confusion matrix."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [val]', leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']

        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = F.interpolate(
                outputs, size=masks.shape[-2:],
                mode='bilinear', align_corners=False,
            )

        loss = criterion(outputs, masks)
        running_loss += loss.item()
        num_batches += 1

        preds = outputs.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = running_loss / max(num_batches, 1)
    ious = compute_iou(all_preds, all_targets, NUM_CLASSES)
    cm = compute_confusion_matrix(all_preds, all_targets, NUM_CLASSES)

    return avg_loss, ious, cm


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: nn.Module,
    output_path: str,
    input_size: int = 256,
    opset_version: int = 12,
    device: torch.device = torch.device('cpu'),
) -> None:
    """Export model to ONNX format."""
    model.eval()
    model = model.to(device)

    # Wrap model to get plain tensor output
    if not isinstance(model, SegmentationModelWrapper):
        wrapper = SegmentationModelWrapper(model)
    else:
        wrapper = model
    wrapper.eval()
    wrapper = wrapper.to(device)

    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    print(f'Exporting ONNX to {output_path}...')
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # Fixed shape for OAK deployment
    )
    print(f'ONNX exported: {output_path}')

    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print('ONNX model validation passed.')


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    val_mious: List[float],
    output_dir: Path,
) -> None:
    """Save loss and mIoU plots."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_mious, label='Val mIoU', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mIoU')
    ax2.set_title('Validation mIoU')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f'Saved training plots to {plot_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build LRASPP-MobileNetV3-Large with custom head."""
    if pretrained:
        model = lraspp_mobilenet_v3_large(weights='DEFAULT')
    else:
        model = lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return model

    # Replace classifier head for our number of classes
    # LRASPP has: classifier.cbr (Conv-BN-ReLU for low features)
    #             classifier.scale (Conv for high features)
    # The final classification is done in the LRASPP head
    low_channels = model.classifier.cbr[0].in_channels
    high_channels = model.classifier.scale[0].in_channels
    inter_channels = model.classifier.cbr[0].out_channels

    # Replace the low-level branch
    model.classifier.cbr = nn.Sequential(
        nn.Conv2d(low_channels, inter_channels, 1, bias=False),
        nn.BatchNorm2d(inter_channels),
        nn.ReLU(inplace=True),
    )

    # Replace the high-level branch
    model.classifier.scale = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(high_channels, inter_channels, 1, bias=False),
        nn.Sigmoid(),
    )

    # Replace final classification layer
    model.classifier.low_classifier = nn.Conv2d(inter_channels, num_classes, 1)
    model.classifier.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train LRASPP-MobileNetV3-Large for plant segmentation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data_dirs', type=str, nargs='+', required=True,
        help='One or more canonical dataset directories',
    )
    parser.add_argument(
        '--output_dir', type=str, default='checkpoints/default',
        help='Directory for checkpoints and logs (default: checkpoints/default)',
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size (default: 8)',
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--input_size', type=int, default=256,
        help='Input image size (default: 256)',
    )
    parser.add_argument(
        '--class_weights', type=float, nargs=3, default=[0.5, 2.0, 3.0],
        metavar=('BG', 'LEAF', 'STEM'),
        help='Loss weights per class (default: 0.5 2.0 3.0)',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--export_onnx', action='store_true',
        help='Export best model to ONNX after training',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader workers (default: 4)',
    )
    parser.add_argument(
        '--no_pretrained', action='store_true',
        help='Train from scratch (no ImageNet pretrained weights)',
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / 'train_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build datasets
    print(f'Loading datasets from: {args.data_dirs}')
    train_datasets = []
    val_datasets = []

    for data_dir in args.data_dirs:
        try:
            train_ds = PlantSegDataset(
                data_dir, split='train',
                input_size=args.input_size, augment=True,
            )
            train_datasets.append(train_ds)
            print(f'  {data_dir} train: {len(train_ds)} samples')
        except FileNotFoundError as e:
            print(f'  WARNING: {e}')

        try:
            val_ds = PlantSegDataset(
                data_dir, split='val',
                input_size=args.input_size, augment=False,
            )
            val_datasets.append(val_ds)
            print(f'  {data_dir} val: {len(val_ds)} samples')
        except FileNotFoundError as e:
            print(f'  WARNING: {e}')

    if not train_datasets:
        print('ERROR: No training data found. Check --data_dirs paths.')
        sys.exit(1)

    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0] if val_datasets else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Build model
    print('Building LRASPP-MobileNetV3-Large...')
    model = build_model(NUM_CLASSES, pretrained=not args.no_pretrained)
    model = model.to(device)

    # Loss with class weights
    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        print(f'Resuming from {args.resume}...')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_miou = checkpoint.get('best_miou', 0.0)
        print(f'  Resumed at epoch {start_epoch}, best mIoU: {best_miou:.4f}')

    # Training loop
    train_losses = []
    val_losses = []
    val_mious = []

    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'  Train samples: {len(train_dataset)}')
    if val_dataset:
        print(f'  Val samples:   {len(val_dataset)}')
    print(f'  Batch size:    {args.batch_size}')
    print(f'  Learning rate: {args.lr}')
    print(f'  Class weights: {args.class_weights}')
    print()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        train_losses.append(train_loss)

        # Validate
        val_loss = 0.0
        ious = {}
        if val_loader:
            val_loss, ious, cm = validate(model, val_loader, criterion, device, epoch + 1)
            val_losses.append(val_loss)
            val_mious.append(ious.get('mIoU', 0.0))

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        elapsed = time.time() - t0
        iou_str = '  '.join(f'{k}: {v:.4f}' for k, v in ious.items()) if ious else 'N/A'
        print(
            f'Epoch {epoch + 1}/{args.epochs}  '
            f'train_loss: {train_loss:.4f}  '
            f'val_loss: {val_loss:.4f}  '
            f'lr: {current_lr:.6f}  '
            f'time: {elapsed:.1f}s'
        )
        if ious:
            print(f'  IoU: {iou_str}')

        # Save best model
        miou = ious.get('mIoU', 0.0)
        if miou > best_miou:
            best_miou = miou
            best_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'class_names': CLASS_NAMES,
                'num_classes': NUM_CLASSES,
                'input_size': args.input_size,
            }, str(best_path))
            print(f'  Saved best model (mIoU: {best_miou:.4f})')

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            latest_path = output_dir / 'latest_checkpoint.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'class_names': CLASS_NAMES,
                'num_classes': NUM_CLASSES,
                'input_size': args.input_size,
            }, str(latest_path))

    # Save final training plots
    if val_mious:
        save_training_plots(train_losses, val_losses, val_mious, output_dir)

    print(f'\nTraining complete. Best mIoU: {best_miou:.4f}')

    # ONNX export
    if args.export_onnx:
        onnx_path = output_dir / 'model.onnx'

        # Load best model
        best_ckpt = torch.load(str(output_dir / 'best_model.pth'),
                               map_location='cpu', weights_only=False)
        model_cpu = build_model(NUM_CLASSES, pretrained=False)
        model_cpu.load_state_dict(best_ckpt['model_state_dict'])

        export_onnx(model_cpu, str(onnx_path), input_size=args.input_size)

        # Copy ONNX to exports/ for convenience
        exports_dir = Path('exports')
        exports_dir.mkdir(exist_ok=True)
        import shutil
        export_copy = exports_dir / 'plant_seg_lraspp.onnx'
        shutil.copy2(str(onnx_path), str(export_copy))
        print(f'ONNX also copied to {export_copy}')


if __name__ == '__main__':
    main()
