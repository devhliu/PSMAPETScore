"""
PSMAPETScore: A Python package for automatic PSMA-PET structure reporting.

This package provides tools for:
1. DICOM I/O handling for PET-CT/MRI
2. Deep learning-based segmentation for CT/MRI
3. PSMA-PET hot lesion segmentation
4. Quantitative analysis based on PROMISE v2 criteria
5. Structured report generation
"""

__version__ = "0.1.0"

from .io import DicomReader, DicomWriter, NiftiConverter
from .segmentation import CTSegmentation, MRISegmentation, PETLesionSegmentation
from .quantification import PROMISECriteria, UptakeAnalysis
from .reporting import ReportGenerator, Visualization
from .models import MODELS_PATH

__all__ = [
    "DicomReader", "DicomWriter", "NiftiConverter",
    "CTSegmentation", "MRISegmentation", "PETLesionSegmentation",
    "PROMISECriteria", "UptakeAnalysis",
    "ReportGenerator", "Visualization",
    "MODELS_PATH"
]