#!/usr/bin/env python3
"""Monitor a training run by watching its output directory.

Usage:
  python scripts/monitor_training.py checkpoints/run08

Polls train_config.yaml and the training log, printing a clean summary table.
Works on Windows without tail -f.
"""

import sys
import time
from pathlib import Path

import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/monitor_training.py <checkpoint_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    log_file = output_dir / "training_log.csv"
    config_file = output_dir / "train_config.yaml"

    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        sys.exit(1)

    # Print config
    if config_file.exists():
        config = yaml.safe_load(config_file.read_text())
        print(f"Run: {output_dir.name}")
        print(f"  Loss:    {config.get('loss_type', 'ce')}")
        print(f"  Weights: {config.get('class_weights', '?')}")
        print(f"  LR:      {config.get('lr', '?')}")
        print(f"  Epochs:  {config.get('epochs', '?')}")
        print()

    print("Waiting for training_log.csv ..." if not log_file.exists() else "")

    last_line_count = 0
    header_printed = False

    while True:
        if not log_file.exists():
            time.sleep(5)
            continue

        lines = log_file.read_text().strip().split("\n")
        if len(lines) <= last_line_count:
            time.sleep(10)
            continue

        # Print header once
        if not header_printed and len(lines) > 0:
            # CSV header
            cols = lines[0].split(",")
            print(f"{'Ep':>4}  {'TrLoss':>7}  {'VaLoss':>7}  "
                  f"{'LR':>10}  {'BG':>6}  {'Leaf':>6}  {'Stem':>6}  {'mIoU':>6}")
            print("-" * 72)
            header_printed = True

        # Print new lines
        for line in lines[max(1, last_line_count):]:
            parts = line.split(",")
            if len(parts) >= 7:
                ep, tl, vl, lr, bg, leaf, stem, miou = parts[:8]
                print(f"{ep:>4}  {float(tl):>7.4f}  {float(vl):>7.4f}  "
                      f"{float(lr):>10.6f}  {float(bg):>6.4f}  {float(leaf):>6.4f}  "
                      f"{float(stem):>6.4f}  {float(miou):>6.4f}")

        last_line_count = len(lines)
        time.sleep(10)


if __name__ == "__main__":
    main()
