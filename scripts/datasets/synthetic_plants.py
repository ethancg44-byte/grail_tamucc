#!/usr/bin/env python3
"""Synthetic Plants dataset adapter.

Converts the Synthetic Plants dataset (Dataset Ninja / Supervisely format)
into the canonical training format.

Expected raw directory structure (Supervisely format):
  raw_dir/
    ds0/  (or similar)
      img/
        image_0001.png
        ...
      ann/
        image_0001.png.json  (Supervisely annotation)
      masks_machine/
        image_0001.png_machine.png  (optional pre-rendered masks)

OR COCO format:
  raw_dir/
    images/
      *.png / *.jpg
    annotations.json   (COCO-format with category info)

Label remapping:
  leaf → 1 (LEAF)
  petiole → 2 (STEM_OR_PETIOLE)
  stem → 2 (STEM_OR_PETIOLE)
  background / unlabeled → 0 (BACKGROUND)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Allow running as script or as module
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.datasets.base_adapter import BaseAdapter, CLASS_MAPPING


# Map source class names to canonical label IDs
LABEL_REMAP = {
    'background': 0,
    'bg': 0,
    '_background_': 0,
    'leaf': 1,
    'leaves': 1,
    'petiole': 2,
    'stem': 2,
    'stems': 2,
    'branch': 2,
}


class SyntheticPlantsAdapter(BaseAdapter):
    """Adapter for Synthetic Plants dataset."""

    @property
    def name(self) -> str:
        return 'SyntheticPlants'

    def convert(self) -> None:
        """Auto-detect format and convert."""
        # Try Supervisely format first
        supervisely_dirs = list(self.raw_dir.glob('*/img'))
        coco_file = self.raw_dir / 'annotations.json'

        if supervisely_dirs:
            print(f'  Detected Supervisely format ({len(supervisely_dirs)} dataset dirs)')
            self._convert_supervisely(supervisely_dirs)
        elif coco_file.exists():
            print('  Detected COCO format')
            self._convert_coco(coco_file)
        else:
            # Try machine masks fallback
            mask_dirs = list(self.raw_dir.glob('*/masks_machine'))
            if mask_dirs:
                print(f'  Detected pre-rendered machine masks ({len(mask_dirs)} dirs)')
                self._convert_machine_masks(mask_dirs)
            else:
                raise FileNotFoundError(
                    f'Could not detect dataset format in {self.raw_dir}. '
                    'Expected Supervisely (*/img/ + */ann/) or COCO (annotations.json) format.'
                )

    def _convert_supervisely(self, img_dirs: List[Path]) -> None:
        """Convert Supervisely-format annotations."""
        count = 0
        for img_dir in sorted(img_dirs):
            ds_dir = img_dir.parent
            ann_dir = ds_dir / 'ann'
            meta_path = ds_dir / 'meta.json'

            # Load class metadata if available
            class_colors = self._load_supervisely_meta(meta_path)

            for img_path in sorted(img_dir.glob('*')):
                if img_path.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                    continue

                ann_path = ann_dir / f'{img_path.name}.json'
                if not ann_path.exists():
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                mask = self._parse_supervisely_annotation(
                    ann_path, img.shape[:2], class_colors
                )

                # Resize
                img_resized = self.resize_image(img, self.input_size)
                mask_resized = self.resize_mask(mask, self.input_size)

                # Save
                name = f'{count:06d}'
                cv2.imwrite(str(self.images_dir / f'{name}.png'), img_resized)
                cv2.imwrite(str(self.masks_dir / f'{name}.png'), mask_resized)
                count += 1

        print(f'  Converted {count} images (Supervisely)')

    def _load_supervisely_meta(self, meta_path: Path) -> Dict[str, int]:
        """Load class-to-label mapping from Supervisely meta.json."""
        class_colors = {}
        if not meta_path.exists():
            return class_colors

        with open(meta_path) as f:
            meta = json.load(f)

        for cls in meta.get('classes', []):
            cls_name = cls.get('title', '').lower()
            if cls_name in LABEL_REMAP:
                class_colors[cls.get('title', '')] = LABEL_REMAP[cls_name]

        return class_colors

    def _parse_supervisely_annotation(
        self,
        ann_path: Path,
        img_shape: Tuple[int, int],
        class_colors: Dict[str, int],
    ) -> np.ndarray:
        """Parse a Supervisely JSON annotation into a label mask."""
        h, w = img_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        with open(ann_path) as f:
            ann = json.load(f)

        for obj in ann.get('objects', []):
            cls_title = obj.get('classTitle', '')
            label_id = class_colors.get(cls_title)

            if label_id is None:
                # Try lowercase lookup
                label_id = LABEL_REMAP.get(cls_title.lower(), None)
            if label_id is None or label_id == 0:
                continue

            geometry_type = obj.get('geometryType', '')

            if geometry_type == 'bitmap':
                self._draw_bitmap(mask, obj, label_id)
            elif geometry_type == 'polygon':
                self._draw_polygon(mask, obj, label_id)

        return mask

    @staticmethod
    def _draw_bitmap(mask: np.ndarray, obj: dict, label_id: int) -> None:
        """Decode a Supervisely bitmap annotation onto the mask."""
        import base64
        import zlib

        bitmap_data = obj.get('bitmap', {})
        data_b64 = bitmap_data.get('data', '')
        origin = bitmap_data.get('origin', [0, 0])

        if not data_b64:
            return

        # Decode base64 → zlib → raw bitmap
        z_data = base64.b64decode(data_b64)
        raw = zlib.decompress(z_data)
        bitmap = np.frombuffer(raw, dtype=np.uint8)

        # Reconstruct bitmap shape from annotation
        h, w = mask.shape
        ox, oy = origin  # origin is [x, y]

        # Bitmap is stored as a flat array of 0/1 values
        bitmap_h = obj.get('bitmap', {}).get('rows', None)
        bitmap_w = obj.get('bitmap', {}).get('cols', None)

        if bitmap_h and bitmap_w:
            bitmap = bitmap.reshape(bitmap_h, bitmap_w)
        else:
            # Try to infer shape from data length
            n_pixels = len(bitmap)
            # bitmap is a packed mask — each byte is 8 pixels
            n_bits = n_pixels * 8
            # This is complex; skip if shape info missing
            return

        # Paint onto mask
        y_end = min(oy + bitmap.shape[0], h)
        x_end = min(ox + bitmap.shape[1], w)
        region = bitmap[:y_end - oy, :x_end - ox]
        mask[oy:y_end, ox:x_end][region > 0] = label_id

    @staticmethod
    def _draw_polygon(mask: np.ndarray, obj: dict, label_id: int) -> None:
        """Draw a polygon annotation onto the mask."""
        points_ext = obj.get('points', {}).get('exterior', [])
        if not points_ext:
            return

        pts = np.array(points_ext, dtype=np.int32)
        cv2.fillPoly(mask, [pts], label_id)

    def _convert_coco(self, ann_file: Path) -> None:
        """Convert COCO-format annotations."""
        with open(ann_file) as f:
            coco = json.load(f)

        # Build category remap
        cat_remap = {}
        for cat in coco.get('categories', []):
            cat_name = cat['name'].lower()
            cat_id = cat['id']
            cat_remap[cat_id] = LABEL_REMAP.get(cat_name, 0)

        # Build image lookup
        images_info = {img['id']: img for img in coco.get('images', [])}

        # Group annotations by image
        anns_by_img: Dict[int, list] = {}
        for ann in coco.get('annotations', []):
            img_id = ann['image_id']
            anns_by_img.setdefault(img_id, []).append(ann)

        # Find images directory
        images_dir = ann_file.parent / 'images'
        if not images_dir.exists():
            images_dir = ann_file.parent

        count = 0
        for img_id, img_info in sorted(images_info.items()):
            file_name = img_info['file_name']
            img_path = images_dir / file_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img_info.get('height', img.shape[0]), img_info.get('width', img.shape[1])
            mask = np.zeros((h, w), dtype=np.uint8)

            for ann in anns_by_img.get(img_id, []):
                label_id = cat_remap.get(ann['category_id'], 0)
                if label_id == 0:
                    continue

                # COCO segmentation can be polygon or RLE
                seg = ann.get('segmentation', [])
                if isinstance(seg, list):
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], label_id)

            img_resized = self.resize_image(img, self.input_size)
            mask_resized = self.resize_mask(mask, self.input_size)

            name = f'{count:06d}'
            cv2.imwrite(str(self.images_dir / f'{name}.png'), img_resized)
            cv2.imwrite(str(self.masks_dir / f'{name}.png'), mask_resized)
            count += 1

        print(f'  Converted {count} images (COCO)')

    def _convert_machine_masks(self, mask_dirs: List[Path]) -> None:
        """Convert pre-rendered machine mask PNGs.

        Machine masks use pixel values as class indices directly.
        We remap them to canonical labels.
        """
        count = 0
        for mask_dir in sorted(mask_dirs):
            ds_dir = mask_dir.parent
            img_dir = ds_dir / 'img'

            for mask_path in sorted(mask_dir.glob('*_machine.png')):
                # Derive image filename
                img_name = mask_path.name.replace('_machine.png', '')
                img_path = img_dir / img_name
                if not img_path.exists():
                    # Try with different extensions
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate = img_dir / (Path(img_name).stem + ext)
                        if candidate.exists():
                            img_path = candidate
                            break
                    else:
                        continue

                img = cv2.imread(str(img_path))
                raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if img is None or raw_mask is None:
                    continue

                # Remap: values in machine mask are typically 0,1,2,...
                # but may not match our canonical mapping — pass through for now
                # (user should verify with dataset documentation)
                mask = np.clip(raw_mask, 0, 2).astype(np.uint8)

                img_resized = self.resize_image(img, self.input_size)
                mask_resized = self.resize_mask(mask, self.input_size)

                name = f'{count:06d}'
                cv2.imwrite(str(self.images_dir / f'{name}.png'), img_resized)
                cv2.imwrite(str(self.masks_dir / f'{name}.png'), mask_resized)
                count += 1

        print(f'  Converted {count} images (machine masks)')


def main():
    parser = argparse.ArgumentParser(
        description='Convert Synthetic Plants dataset to canonical training format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Supervisely format
  python synthetic_plants.py --raw_dir data/synthetic_plants_raw --output_dir data/synthetic_plants

  # COCO format
  python synthetic_plants.py --raw_dir data/synthetic_plants_coco --output_dir data/synthetic_plants

  # Custom split ratios
  python synthetic_plants.py --raw_dir data/raw --output_dir data/converted --split 0.7 0.15 0.15
        """,
    )
    parser.add_argument(
        '--raw_dir', type=str, required=True,
        help='Path to raw Synthetic Plants dataset directory',
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

    adapter = SyntheticPlantsAdapter(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        input_size=args.input_size,
        split_ratios=tuple(args.split),
        seed=args.seed,
    )
    adapter.run()


if __name__ == '__main__':
    main()
