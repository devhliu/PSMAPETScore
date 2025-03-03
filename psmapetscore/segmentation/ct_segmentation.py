"""
CT segmentation using nnUNetv2.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_segmentation import BaseSegmentation
from ..models import CT_SEGMENTATION_MODEL


class CTSegmentation(BaseSegmentation):
    """
    Class for CT whole-body segmentation using nnUNetv2.
    """
    
    def __init__(self, model_folder: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize CT segmentation model.
        
        Args:
            model_folder: Path to the trained model folder (defaults to package model)
            use_gpu: Whether to use GPU for inference
        """
        if model_folder is None:
            model_folder = str(CT_SEGMENTATION_MODEL)
            
        super().__init__(
            model_folder=model_folder,
            task_name="Task503_CTWholebody",  # Example task name
            use_gpu=use_gpu
        )
        
        self.organ_labels = {
            1: "Liver",
            2: "Spleen",
            3: "Kidney_R",
            4: "Kidney_L",
            5: "Prostate",
            6: "Bladder",
            7: "Lung_R",
            8: "Lung_L",
            9: "Blood_Pool",
            10: "Bone"
        }
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess CT image.
        
        Args:
            image: Input CT image array
            
        Returns:
            Preprocessed image array
        """
        # Apply CT window/level
        image = np.clip(image, -1000, 1000)
        image = (image + 1000) / 2000  # Normalize to [0, 1]
        return image
    
    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """
        Postprocess the segmentation prediction.
        
        Args:
            prediction: Raw prediction from nnUNet
            
        Returns:
            Processed segmentation mask
        """
        # Convert probabilities to labels
        if len(prediction.shape) == 4:  # If probabilities
            prediction = np.argmax(prediction, axis=0)
        return prediction.astype(np.uint8)