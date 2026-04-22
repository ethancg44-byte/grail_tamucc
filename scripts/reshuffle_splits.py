#!/usr/bin/env python3
"""Reshuffle dataset splits (e.g., 80/10/10 -> 90/5/5).

Reads all sample names from existing split files, reshuffles with a
deterministic seed, and writes new split files. Original splits are
backed up to splits/backup_YYYYMMDD_HHMMSS/.

Usage:
  python reshuffle_splits.py --data_dir data/synthetic_plants --split 0.9 0.05 0.05
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Reshuffle dataset splits.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to canonical dataset directory')
    parser.add_argument('--split', type=float, nargs=3, default=[0.9, 0.05, 0.05],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='New split ratios (default: 0.9 0.05 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    splits_dir = Path(args.data_dir) / 'splits'
    if not splits_dir.exists():
        print(f'ERROR: splits directory not found: {splits_dir}')
        return

    # Collect all sample names from existing splits
    all_names = []
    for split_file in ['train.txt', 'val.txt', 'test.txt']:
        path = splits_dir / split_file
        if path.exists():
            names = [l.strip() for l in path.read_text().strip().split('\n') if l.strip()]
            all_names.extend(names)
            print(f'  Read {len(names)} samples from {split_file}')

    total = len(all_names)
    print(f'Total samples: {total}')

    # Backup existing splits
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    backup_dir = splits_dir / f'backup_{timestamp}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in splits_dir.glob('*.txt'):
        shutil.copy2(str(f), str(backup_dir / f.name))
    print(f'Backed up existing splits to {backup_dir}')

    # Shuffle and split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(total)

    train_ratio, val_ratio, test_ratio = args.split
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    # Remainder goes to test
    n_test = total - n_train - n_val

    train_names = [all_names[i] for i in indices[:n_train]]
    val_names = [all_names[i] for i in indices[n_train:n_train + n_val]]
    test_names = [all_names[i] for i in indices[n_train + n_val:]]

    # Write new splits
    for split_name, names in [('train', train_names), ('val', val_names), ('test', test_names)]:
        path = splits_dir / f'{split_name}.txt'
        path.write_text('\n'.join(sorted(names)) + '\n')
        print(f'  Wrote {len(names)} samples to {split_name}.txt')

    print(f'\nNew split: {len(train_names)}/{len(val_names)}/{len(test_names)} '
          f'({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})')


if __name__ == '__main__':
    main()
