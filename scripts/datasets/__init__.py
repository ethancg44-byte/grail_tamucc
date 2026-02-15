"""Dataset adapters for plant segmentation training.

Each adapter normalizes a specific dataset into a common format:
  images/*.png  — RGB images resized to target resolution
  masks/*.png   — mono8 label masks (0=BG, 1=LEAF, 2=STEM_OR_PETIOLE)
  labels.yaml   — class mapping metadata
  splits/       — train.txt, val.txt, test.txt with filenames
"""

from scripts.datasets.synthetic_plants import SyntheticPlantsAdapter
from scripts.datasets.cvppp import CVPPPAdapter

__all__ = ['SyntheticPlantsAdapter', 'CVPPPAdapter']
