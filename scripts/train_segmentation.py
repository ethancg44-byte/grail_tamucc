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
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large,
    lraspp_mobilenet_v3_large,
)
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
        aug_level: str = 'basic',
    ):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augment = augment
        self.aug_level = aug_level

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
            if self.aug_level == 'strong':
                img, mask = self._augment_strong(img, mask)
            else:
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

    @staticmethod
    def _augment_strong(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Strong augmentation pipeline for better generalization."""
        h, w = img.shape[:2]

        # --- Geometric augmentations (applied to both image and mask) ---

        # Random horizontal flip
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Random continuous rotation (-45 to +45 degrees)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-45, 45)
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_REFLECT_101)

        # Random scale and crop (0.8x - 1.2x zoom)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            new_h, new_w = int(h * scale), int(w * scale)
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Crop or pad back to original size
            if scale > 1.0:
                y0 = np.random.randint(0, new_h - h + 1)
                x0 = np.random.randint(0, new_w - w + 1)
                img = img_scaled[y0:y0+h, x0:x0+w]
                mask = mask_scaled[y0:y0+h, x0:x0+w]
            else:
                # Pad with reflect
                pad_y = h - new_h
                pad_x = w - new_w
                y0 = pad_y // 2
                x0 = pad_x // 2
                img_pad = np.zeros((h, w, 3), dtype=img.dtype)
                mask_pad = np.zeros((h, w), dtype=mask.dtype)
                img_pad[y0:y0+new_h, x0:x0+new_w] = img_scaled
                mask_pad[y0:y0+new_h, x0:x0+new_w] = mask_scaled
                # Fill borders with reflection
                img = cv2.copyMakeBorder(img_scaled,
                                         y0, pad_y - y0, x0, pad_x - x0,
                                         cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask_scaled,
                                          y0, pad_y - y0, x0, pad_x - x0,
                                          cv2.BORDER_REFLECT_101)

        # Elastic deformation (biggest win for thin structures like stems)
        if np.random.random() > 0.5:
            alpha = np.random.uniform(80, 120)  # displacement magnitude
            sigma = np.random.uniform(9, 11)    # smoothness
            dx = gaussian_filter(np.random.randn(h, w), sigma) * alpha
            dy = gaussian_filter(np.random.randn(h, w), sigma) * alpha
            y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            indices_y = np.clip(y_grid + dy, 0, h - 1)
            indices_x = np.clip(x_grid + dx, 0, w - 1)
            for c in range(3):
                img[:, :, c] = map_coordinates(img[:, :, c],
                                               [indices_y, indices_x],
                                               order=1, mode='reflect')
            mask = map_coordinates(mask, [indices_y, indices_x],
                                   order=0, mode='reflect')

        # --- Photometric augmentations (image only) ---

        # Random brightness/contrast
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.7, 1.3)  # contrast
            beta = np.random.uniform(-30, 30)     # brightness
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Random color jitter
        if np.random.random() > 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-15, 15)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Gaussian blur
        if np.random.random() > 0.3:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # Gaussian noise
        if np.random.random() > 0.5:
            noise_std = np.random.uniform(5, 25)
            noise = np.random.randn(*img.shape) * noise_std
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random erasing / cutout (mask-aware: only erase if region is mostly BG)
        if np.random.random() > 0.5:
            er_h = np.random.randint(h // 8, h // 4)
            er_w = np.random.randint(w // 8, w // 4)
            er_y = np.random.randint(0, h - er_h)
            er_x = np.random.randint(0, w - er_w)
            # Fill with mean color
            mean_color = img.mean(axis=(0, 1)).astype(np.uint8)
            img[er_y:er_y+er_h, er_x:er_x+er_w] = mean_color

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
# Loss Functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for semantic segmentation.

    Computes per-class Dice coefficient and returns 1 - mean(Dice).
    Naturally handles class imbalance since each class contributes equally.
    """

    def __init__(self, num_classes: int = 3, smooth: float = 1.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.weight = weight  # per-class weight tensor

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_oh = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets_oh.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.weight is not None:
            dice_score = dice_score * self.weight.to(dice_score.device)
            return 1.0 - dice_score.sum() / self.weight.sum()

        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    """Focal Loss for semantic segmentation.

    Down-weights easy (well-classified) pixels so the model focuses on hard ones.
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    """

    def __init__(self, gamma: float = 2.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight,
                                  reduction='none')  # (B, H, W)
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        # Gather the probability of the true class for each pixel
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H, W)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


class CombinedLoss(nn.Module):
    """CE + Dice combined loss.

    CE provides stable early-training gradients; Dice pushes per-class overlap.
    loss = ce_weight * CE + dice_weight * Dice
    """

    def __init__(self, num_classes: int = 3, ce_weight: float = 1.0,
                 dice_weight: float = 1.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(num_classes=num_classes, weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.ce_weight * self.ce(logits, targets)
                + self.dice_weight * self.dice(logits, targets))


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax loss for semantic segmentation.

    Directly optimizes the mean IoU metric via its convex surrogate
    (Lovász extension of the Jaccard index). Based on:
    Berman et al., "The Lovász-Softmax loss" (CVPR 2018).
    """

    def __init__(self, num_classes: int = 3,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight  # per-class weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        losses = []
        for c in range(self.num_classes):
            fg = (targets == c).float()  # (B, H, W) binary ground truth
            errors = (fg - probs[:, c]).abs()  # per-pixel errors
            fg_sorted, perm = errors.view(-1).sort(descending=True)
            fg_target = fg.view(-1)[perm]
            # Lovász extension: compute IoU gradient
            intersection = fg_target.cumsum(0)
            union = torch.arange(1, len(fg_target) + 1,
                                 device=fg_target.device).float() + intersection - fg_target
            jaccard = 1.0 - intersection / union
            # Compute gradient of Lovász extension
            jaccard_grad = torch.zeros_like(jaccard)
            jaccard_grad[0] = jaccard[0]
            jaccard_grad[1:] = jaccard[1:] - jaccard[:-1]
            loss_c = (fg_sorted * jaccard_grad).sum()
            if self.weight is not None:
                loss_c = loss_c * self.weight[c].to(loss_c.device)
            losses.append(loss_c)
        if self.weight is not None:
            return sum(losses) / self.weight.sum()
        return sum(losses) / self.num_classes


class TverskyLoss(nn.Module):
    """Tversky loss for semantic segmentation.

    Generalizes Dice loss with asymmetric FN/FP penalties:
    TI = TP / (TP + alpha*FN + beta*FP)
    Higher alpha penalizes missed predictions (false negatives) more,
    which helps recall thin structures like stems.
    """

    def __init__(self, num_classes: int = 3, alpha: float = 0.7,
                 beta: float = 0.3, smooth: float = 1.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)
        targets_oh = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)  # sum over batch, H, W
        tp = (probs * targets_oh).sum(dim=dims)
        fn = (targets_oh * (1 - probs)).sum(dim=dims)
        fp = ((1 - targets_oh) * probs).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)

        if self.weight is not None:
            tversky = tversky * self.weight.to(tversky.device)
            return 1.0 - tversky.sum() / self.weight.sum()

        return 1.0 - tversky.mean()


class CombinedCETverskyLoss(nn.Module):
    """CE + Tversky combined loss.

    CE provides stable early-training gradients; Tversky pushes
    recall on minority classes (stems) via asymmetric FN/FP penalty.
    """

    def __init__(self, num_classes: int = 3, ce_weight_factor: float = 1.0,
                 tversky_weight: float = 1.0, alpha: float = 0.7,
                 beta: float = 0.3,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce_weight_factor = ce_weight_factor
        self.tversky_weight = tversky_weight
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.tversky = TverskyLoss(num_classes=num_classes, alpha=alpha,
                                   beta=beta, weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.ce_weight_factor * self.ce(logits, targets)
                + self.tversky_weight * self.tversky(logits, targets))


class CombinedCELovaszLoss(nn.Module):
    """CE + Lovász-Softmax combined loss.

    CE provides stable gradients in early training; Lovász directly
    optimizes IoU once predictions are reasonable.
    """

    def __init__(self, num_classes: int = 3, ce_weight_factor: float = 1.0,
                 lovasz_weight: float = 1.0,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce_weight_factor = ce_weight_factor
        self.lovasz_weight = lovasz_weight
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.lovasz = LovaszSoftmaxLoss(num_classes=num_classes, weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.ce_weight_factor * self.ce(logits, targets)
                + self.lovasz_weight * self.lovasz(logits, targets))


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

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

        preds = outputs.argmax(dim=1).cpu().numpy().astype(np.int32)
        targets = masks.cpu().numpy().astype(np.int32)
        for t in range(NUM_CLASSES):
            for p in range(NUM_CLASSES):
                cm[t, p] += np.sum((targets == t) & (preds == p))

    avg_loss = running_loss / max(num_batches, 1)

    # Derive per-class IoU from confusion matrix
    ious = {}
    for cls in range(NUM_CLASSES):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        union = tp + fn + fp
        if union == 0:
            ious[CLASS_NAMES[cls]] = float('nan')
        else:
            ious[CLASS_NAMES[cls]] = float(tp) / float(union)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious['mIoU'] = float(np.mean(valid_ious)) if valid_ious else 0.0

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

def build_model(num_classes: int, pretrained: bool = True,
                arch: str = 'lraspp') -> nn.Module:
    """Build segmentation model with custom head.

    Args:
        arch: 'lraspp' for LRASPP-MobileNetV3-Large (lightweight),
              'deeplabv3' for DeepLabV3-MobileNetV3-Large (richer decoder).
    """
    if arch == 'deeplabv3':
        if pretrained:
            model = deeplabv3_mobilenet_v3_large(weights='DEFAULT')
        else:
            model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=num_classes)
            return model
        # DeepLabV3 classifier: Sequential(..., Conv2d(256, num_classes, 1))
        # Replace only the final conv layer to keep pretrained ASPP weights.
        in_ch = model.classifier[-1].in_channels  # 256
        model.classifier[-1] = nn.Conv2d(in_ch, num_classes, 1)
        return model

    # Default: LRASPP
    if pretrained:
        model = lraspp_mobilenet_v3_large(weights='DEFAULT')
    else:
        model = lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
        return model

    # Replace only the final classification layers for our number of classes.
    low_in = model.classifier.low_classifier.in_channels   # 40
    high_in = model.classifier.high_classifier.in_channels  # 128
    model.classifier.low_classifier = nn.Conv2d(low_in, num_classes, 1)
    model.classifier.high_classifier = nn.Conv2d(high_in, num_classes, 1)
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
        '--loss_type', type=str, default='ce',
        choices=['ce', 'dice', 'focal', 'ce_dice', 'lovasz', 'ce_lovasz',
                 'tversky', 'ce_tversky'],
        help='Loss function (default: ce)',
    )
    parser.add_argument(
        '--focal_gamma', type=float, default=2.0,
        help='Focal loss gamma parameter (default: 2.0)',
    )
    parser.add_argument(
        '--tversky_alpha', type=float, default=0.7,
        help='Tversky alpha (FN penalty weight, default: 0.7)',
    )
    parser.add_argument(
        '--tversky_beta', type=float, default=0.3,
        help='Tversky beta (FP penalty weight, default: 0.3)',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from',
    )
    parser.add_argument(
        '--init_weights', type=str, default=None,
        help='Load model weights only (no optimizer/scheduler/epoch). For fine-tuning with a new loss.',
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
    parser.add_argument(
        '--model', type=str, default='lraspp',
        choices=['lraspp', 'deeplabv3'],
        help='Model architecture: lraspp (light) or deeplabv3 (richer decoder) (default: lraspp)',
    )
    parser.add_argument(
        '--aug_level', type=str, default='strong',
        choices=['basic', 'strong'],
        help='Augmentation level: basic (flips/rot90) or strong (default: strong)',
    )
    parser.add_argument(
        '--stem_sampling', action='store_true',
        help='Enable stem-focused oversampling (WeightedRandomSampler by stem pixel ratio)',
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=5,
        help='LR warmup epochs before cosine decay (default: 5)',
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
                aug_level=args.aug_level,
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

    # Stem-focused oversampling
    sampler = None
    shuffle = True
    if args.stem_sampling:
        print('Computing stem pixel ratios for weighted sampling...')
        stem_ratios = []
        for ds in train_datasets:
            for name in ds.sample_names:
                mask_path = ds.data_dir / 'masks' / f'{name}.png'
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    stem_pixels = np.sum(mask == 2)
                    total_pixels = mask.size
                    stem_ratios.append(stem_pixels / total_pixels)
                else:
                    stem_ratios.append(0.0)
        # Weight = sqrt(stem_ratio) so stem-heavy images are sampled more often
        # Add small epsilon so images with no stem still get sampled occasionally
        weights = [np.sqrt(r) + 0.01 for r in stem_ratios]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        shuffle = False  # mutually exclusive with sampler
        mean_ratio = np.mean(stem_ratios)
        max_ratio = np.max(stem_ratios)
        print(f'  Stem ratios: mean={mean_ratio:.4f}, max={max_ratio:.4f}')
        print(f'  Weighted sampler created ({len(weights)} samples)')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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
    print(f'Building {args.model.upper()}-MobileNetV3-Large...')
    model = build_model(NUM_CLASSES, pretrained=not args.no_pretrained, arch=args.model)
    model = model.to(device)

    # Loss function
    class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss_type == 'dice':
        criterion = DiceLoss(num_classes=NUM_CLASSES, weight=class_weights)
    elif args.loss_type == 'focal':
        criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    elif args.loss_type == 'ce_dice':
        criterion = CombinedLoss(num_classes=NUM_CLASSES, weight=class_weights)
    elif args.loss_type == 'lovasz':
        criterion = LovaszSoftmaxLoss(num_classes=NUM_CLASSES, weight=class_weights)
    elif args.loss_type == 'ce_lovasz':
        criterion = CombinedCELovaszLoss(num_classes=NUM_CLASSES, weight=class_weights)
    elif args.loss_type == 'tversky':
        criterion = TverskyLoss(num_classes=NUM_CLASSES, alpha=args.tversky_alpha,
                                beta=args.tversky_beta, weight=class_weights)
    elif args.loss_type == 'ce_tversky':
        criterion = CombinedCETverskyLoss(num_classes=NUM_CLASSES,
                                          alpha=args.tversky_alpha,
                                          beta=args.tversky_beta,
                                          weight=class_weights)
    print(f'Loss function: {args.loss_type} ({criterion.__class__.__name__})')

    # Optimizer and scheduler (warmup + cosine decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    cosine_epochs = max(args.epochs - warmup_epochs, 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

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
    elif args.init_weights:
        print(f'Loading model weights from {args.init_weights}...')
        checkpoint = torch.load(args.init_weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        src_miou = checkpoint.get('best_miou', '?')
        src_epoch = checkpoint.get('epoch', '?')
        print(f'  Loaded weights from epoch {src_epoch} (mIoU: {src_miou}). Fresh optimizer/scheduler.')

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
    print(f'  Model:         {args.model}')
    print(f'  Class weights: {args.class_weights}')
    print(f'  Loss type:     {args.loss_type}')
    print(f'  Aug level:     {args.aug_level}')
    print(f'  Stem sampling: {args.stem_sampling}')
    print(f'  Warmup epochs: {warmup_epochs}')
    print(f'  Grad clipping: max_norm=1.0')
    print()

    # CSV log for monitoring
    log_path = output_dir / 'training_log.csv'
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,lr,bg_iou,leaf_iou,stem_iou,miou\n')

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

        # Append to CSV log
        with open(log_path, 'a') as f:
            f.write(f'{epoch + 1},{train_loss:.6f},{val_loss:.6f},{current_lr:.8f},'
                    f'{ious.get("BACKGROUND", 0.0):.6f},'
                    f'{ious.get("LEAF", 0.0):.6f},'
                    f'{ious.get("STEM_OR_PETIOLE", 0.0):.6f},'
                    f'{ious.get("mIoU", 0.0):.6f}\n')

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
                'arch': args.model,
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
                'arch': args.model,
            }, str(latest_path))

    # Save final training plots
    if val_mious:
        save_training_plots(train_losses, val_losses, val_mious, output_dir)

    print(f'\nTraining complete. Best mIoU: {best_miou:.4f}')

    # ONNX export
    if args.export_onnx:
        try:
            onnx_path = output_dir / 'model.onnx'

            # Load best model
            best_ckpt = torch.load(str(output_dir / 'best_model.pth'),
                                   map_location='cpu', weights_only=False)
            model_cpu = build_model(NUM_CLASSES, pretrained=False, arch=args.model)
            model_cpu.load_state_dict(best_ckpt['model_state_dict'])

            export_onnx(model_cpu, str(onnx_path), input_size=args.input_size)

            # Copy ONNX to exports/ for convenience
            exports_dir = Path('exports')
            exports_dir.mkdir(exist_ok=True)
            import shutil
            export_copy = exports_dir / 'plant_seg_lraspp.onnx'
            shutil.copy2(str(onnx_path), str(export_copy))
            print(f'ONNX also copied to {export_copy}')
        except Exception as e:
            print(f'\nERROR: ONNX export failed: {e}')
            print('Training results are saved. You can retry export manually.')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
