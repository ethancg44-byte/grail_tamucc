#!/usr/bin/env python3
"""CVPPP Leaf Segmentation Challenge (LSC) dataset adapter.

Converts CVPPP dataset into canonical training format.

Expected raw directory structure:
  raw_dir/
    A1/  (and/or A2, A3, A4 — different plant species)
      plant001_rgb.png
      plant001_label.png    (instance mask: each leaf = unique integer)
      plant001_fg.png       (optional: binary foreground mask)
      ...

Label remapping:
  - Instance labels (1,2,3,...) → LEAF (1)
  - Background (0) → BACKGROUND (0)
  - No stem class available — warns user

NOTE: CVPPP only provides leaf labels. Stem/petiole (class 2) will NOT be
present. Use this dataset to bootstrap leaf segmentation, then combine with
Synthetic Plants for full 3-class training.
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.datasets.base_adapter import BaseAdapter


class CVPPPAdapter(BaseAdapter):
    """Adapter for CVPPP Leaf Segmentation Challenge dataset."""

    @property
    def name(self) -> str:
        return 'CVPPP_LSC'

    def convert(self) -> None:
        """Convert CVPPP instance masks to semantic labels."""
        warnings.warn(
            'CVPPP dataset contains LEAF labels only (no stem/petiole). '
            'Class 2 (STEM_OR_PETIOLE) will not be present in this dataset. '
            'Combine with Synthetic Plants for full 3-class training.',
            stacklevel=2,
        )

        # Find all subdirectories (A1, A2, A3, A4)
        subdirs = self._find_data_dirs()
        if not subdirs:
            raise FileNotFoundError(
                f'No data directories found in {self.raw_dir}. '
                'Expected subdirectories like A1/, A2/, etc. with '
                'plant*_rgb.png and plant*_label.png files.'
            )

        count = 0
        for subdir in subdirs:
            print(f'  Processing {subdir.name}...')
            rgb_files = sorted(subdir.glob('*_rgb.png'))

            for rgb_path in rgb_files:
                # Derive label path
                stem = rgb_path.name.replace('_rgb.png', '')
                label_path = subdir / f'{stem}_label.png'

                if not label_path.exists():
                    continue

                img = cv2.imread(str(rgb_path))
                instance_mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)

                if img is None or instance_mask is None:
                    continue

                # Handle multi-channel label images
                if len(instance_mask.shape) == 3:
                    # Some CVPPP labels are stored as RGB — use first channel
                    # or convert to grayscale
                    instance_mask = instance_mask[:, :, 0]

                # Convert instance mask to semantic mask:
                # any non-zero value → LEAF (1), zero → BACKGROUND (0)
                semantic_mask = np.zeros_like(instance_mask, dtype=np.uint8)
                semantic_mask[instance_mask > 0] = 1  # LEAF

                # Resize
                img_resized = self.resize_image(img, self.input_size)
                mask_resized = self.resize_mask(semantic_mask, self.input_size)

                name = f'{count:06d}'
                cv2.imwrite(str(self.images_dir / f'{name}.png'), img_resized)
                cv2.imwrite(str(self.masks_dir / f'{name}.png'), mask_resized)
                count += 1

        print(f'  Converted {count} images (CVPPP)')
        if count == 0:
            print('  WARNING: No images converted. Check raw directory structure.')

    def _find_data_dirs(self) -> List[Path]:
        """Find CVPPP data subdirectories."""
        # Standard CVPPP layout: A1/, A2/, A3/, A4/
        dirs = []
        for name in ['A1', 'A2', 'A3', 'A4']:
            d = self.raw_dir / name
            if d.is_dir():
                dirs.append(d)

        # Also check for flat layout (rgb+label files directly in raw_dir)
        if not dirs:
            if list(self.raw_dir.glob('*_rgb.png')):
                dirs.append(self.raw_dir)

        # Also check one level deeper (e.g., raw_dir/training/A1/)
        if not dirs:
            for subdir in sorted(self.raw_dir.iterdir()):
                if subdir.is_dir():
                    for name in ['A1', 'A2', 'A3', 'A4']:
                        d = subdir / name
                        if d.is_dir():
                            dirs.append(d)

        return dirs


def main():
    parser = argparse.ArgumentParser(
        description='Convert CVPPP LSC dataset to canonical training format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NOTE: CVPPP only provides leaf instance masks (no stem/petiole labels).
      The converted dataset will only contain classes 0 (BG) and 1 (LEAF).
      Combine with Synthetic Plants dataset for full 3-class training.

Examples:
  python cvppp.py --raw_dir data/cvppp_raw --output_dir data/cvppp

  # Process only A1 subset
  python cvppp.py --raw_dir data/cvppp_raw/A1 --output_dir data/cvppp_a1

  # Custom resolution
  python cvppp.py --raw_dir data/cvppp_raw --output_dir data/cvppp --input_size 512
        """,
    )
    parser.add_argument(
        '--raw_dir', type=str, required=True,
        help='Path to raw CVPPP dataset directory',
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Path to output directory for converted dataset',
    )
    parser.add_argument(
        '--input_size', type=int, default=256,
        help='Target image size (default: 256)',
    )
    parser.add_argument(
        '--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Split ratios for train/val/test (default: 0.8 0.1 0.1)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for split shuffling (default: 42)',
    )

    args = parser.parse_args()

    adapter = CVPPPAdapter(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        input_size=args.input_size,
        split_ratios=tuple(args.split),
        seed=args.seed,
    )
    adapter.run()


if __name__ == '__main__':
    main()
