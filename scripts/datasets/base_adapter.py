"""Abstract base class for dataset adapters.

Every adapter converts a raw dataset into a normalized format:
  output_dir/
    images/*.png   — RGB images resized to (input_size x input_size)
    masks/*.png    — mono8 label masks (0=BG, 1=LEAF, 2=STEM_OR_PETIOLE)
    labels.yaml    — class mapping metadata
    splits/
      train.txt    — one filename per line (no extension)
      val.txt
      test.txt
"""

import abc
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml


# Canonical class mapping shared by all adapters
CLASS_MAPPING = {
    0: 'BACKGROUND',
    1: 'LEAF',
    2: 'STEM_OR_PETIOLE',
}


class BaseAdapter(abc.ABC):
    """Abstract dataset adapter."""

    def __init__(
        self,
        raw_dir: str,
        output_dir: str,
        input_size: int = 256,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.input_size = input_size
        self.split_ratios = split_ratios
        self.seed = seed

        # Output subdirectories
        self.images_dir = self.output_dir / 'images'
        self.masks_dir = self.output_dir / 'masks'
        self.splits_dir = self.output_dir / 'splits'

    def run(self) -> None:
        """Execute the full conversion pipeline."""
        print(f'[{self.name}] Starting conversion...')
        print(f'  Raw dir:    {self.raw_dir}')
        print(f'  Output dir: {self.output_dir}')
        print(f'  Input size: {self.input_size}x{self.input_size}')

        self._create_dirs()
        self.convert()
        self.create_splits()
        self._write_labels_yaml()
        self.validate()
        print(f'[{self.name}] Done.')

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable dataset name."""

    @abc.abstractmethod
    def convert(self) -> None:
        """Convert raw dataset files into images/*.png + masks/*.png.

        Subclasses must:
        1. Read raw images and annotations from self.raw_dir
        2. Remap annotations to canonical labels (0=BG, 1=LEAF, 2=STEM)
        3. Resize both to self.input_size x self.input_size
        4. Save to self.images_dir and self.masks_dir as PNG
        """

    def create_splits(self) -> None:
        """Create train/val/test splits from converted images."""
        all_files = sorted(
            f.stem for f in self.images_dir.glob('*.png')
        )
        if not all_files:
            raise RuntimeError(
                f'No images found in {self.images_dir}. '
                'Did convert() run successfully?'
            )

        random.seed(self.seed)
        random.shuffle(all_files)

        n = len(all_files)
        n_train = int(n * self.split_ratios[0])
        n_val = int(n * self.split_ratios[1])

        splits = {
            'train': all_files[:n_train],
            'val': all_files[n_train:n_train + n_val],
            'test': all_files[n_train + n_val:],
        }

        for split_name, filenames in splits.items():
            split_file = self.splits_dir / f'{split_name}.txt'
            split_file.write_text('\n'.join(filenames) + '\n')
            print(f'  {split_name}: {len(filenames)} samples')

    def validate(self) -> None:
        """Verify that the converted dataset is well-formed."""
        errors = []

        # Check images and masks match
        image_stems = {f.stem for f in self.images_dir.glob('*.png')}
        mask_stems = {f.stem for f in self.masks_dir.glob('*.png')}

        missing_masks = image_stems - mask_stems
        if missing_masks:
            errors.append(
                f'{len(missing_masks)} images missing masks: '
                f'{list(missing_masks)[:5]}...'
            )

        extra_masks = mask_stems - image_stems
        if extra_masks:
            errors.append(
                f'{len(extra_masks)} masks without images: '
                f'{list(extra_masks)[:5]}...'
            )

        # Check mask label values
        for mask_file in sorted(self.masks_dir.glob('*.png'))[:20]:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                errors.append(f'Could not read mask: {mask_file.name}')
                continue
            unique_vals = set(np.unique(mask))
            invalid = unique_vals - {0, 1, 2}
            if invalid:
                errors.append(
                    f'{mask_file.name}: invalid label values {invalid}'
                )

        # Check split files reference existing images
        for split_file in self.splits_dir.glob('*.txt'):
            names = split_file.read_text().strip().split('\n')
            for name in names:
                if name and name not in image_stems:
                    errors.append(
                        f'{split_file.name}: references missing image "{name}"'
                    )
                    break  # one error per split is enough

        if errors:
            print(f'  VALIDATION WARNINGS ({len(errors)}):')
            for err in errors:
                print(f'    - {err}')
        else:
            print(f'  Validation passed: {len(image_stems)} image/mask pairs OK')

    def _create_dirs(self) -> None:
        """Create output directory structure."""
        for d in [self.images_dir, self.masks_dir, self.splits_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _write_labels_yaml(self) -> None:
        """Write class mapping metadata."""
        labels = {
            'num_classes': len(CLASS_MAPPING),
            'classes': CLASS_MAPPING,
            'source_dataset': self.name,
            'input_size': self.input_size,
        }
        labels_path = self.output_dir / 'labels.yaml'
        with open(labels_path, 'w') as f:
            yaml.dump(labels, f, default_flow_style=False)
        print(f'  Wrote {labels_path}')

    @staticmethod
    def resize_image(img: np.ndarray, size: int) -> np.ndarray:
        """Resize image preserving aspect via center crop + resize."""
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
        """Resize mask using nearest-neighbor to preserve label values."""
        return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
