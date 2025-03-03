"""
PSMA-PET lesion segmentation using nnUNetv2.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_segmentation import BaseSegmentation
from ..models import PET_LESION_MODEL


class PETLesionSegmentation(BaseSegmentation):
    """
    Class for PSMA-PET lesion segmentation using nnUNetv2.
    """
    
    def __init__(self, model_folder: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize PET lesion segmentation model.
        
        Args:
            model_folder: Path to the trained model folder (defaults to package model)
            use_gpu: Whether to use GPU for inference
        """
        if model_folder is None:
            model_folder = str(PET_LESION_MODEL)
            
        super().__init__(
            model_folder=model_folder,
            task_name="Task505_PSMALesion",  # Example task name
            use_gpu=use_gpu
        )
        
        self.lesion_labels = {
            1: "PSMA_Lesion"
        }
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess PET image.
        
        Args:
            image: Input PET SUV image array
            
        Returns:
            Preprocessed image array
        """
        # Clip and normalize SUV values
        image = np.clip(image, 0, 20)  # Clip SUV to [0, 20]
        image = image / 20.0  # Normalize to [0, 1]
        return image
    
    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """
        Postprocess the segmentation prediction.
        
        Args:
            prediction: Raw prediction from nnUNet
            
        Returns:
            Processed segmentation mask
        """
        # Convert probabilities to binary mask
        if len(prediction.shape) == 4:  # If probabilities
            prediction = (prediction[1] > 0.5).astype(np.uint8)  # Use class 1 probability
        return prediction