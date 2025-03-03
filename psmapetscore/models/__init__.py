"""
Pre-trained models for PSMA-PET analysis.
"""

import os
from pathlib import Path

# Define the base path for models
MODELS_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

# Model directories
CT_SEGMENTATION_MODEL = MODELS_PATH / "ct_segmentation"
MRI_SEGMENTATION_MODEL = MODELS_PATH / "mri_segmentation"
PET_LESION_MODEL = MODELS_PATH / "pet_lesion"

__all__ = ["MODELS_PATH", "CT_SEGMENTATION_MODEL", "MRI_SEGMENTATION_MODEL", "PET_LESION_MODEL"]