"""
MRI segmentation using nnUNetv2.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .base_segmentation import BaseSegmentation
from ..models import MRI_SEGMENTATION_MODEL


class MRISegmentation(BaseSegmentation):
    """
    Class for MRI whole-body segmentation using nnUNetv2.
    """
    
    def __init__(self, model_folder: Optional[str] = None, use_gpu: bool = True):
        """
        Initialize MRI segmentation model.
        
        Args:
            model_folder: Path to the trained model folder (defaults to package model)
            use_gpu: Whether to use GPU for inference
        """
        if model_folder is None:
            model_folder = str(MRI_SEGMENTATION_MODEL)
            
        super().__init__(
            model_folder=model_folder,
            task_name="Task504_MRIWholebody",  # Example task name
            use_gpu=use_gpu
        )
        
        self.organ_labels = {
            1: "Liver",
            2: "Spleen",
            3: "Kidney_R",
            4: "Kidney_L",
            5: "Prostate",
            6: "Bladder",
            7: "Bone",
            8: "Muscle",
            9: "Blood_Pool",
            10: "Fat"
        }
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess MRI image.
        
        Args:
            image: Input MRI image array
            
        Returns:
            Preprocessed image array
        """
        # Normalize to zero mean and unit variance
        mean = np.mean(image)
        std = np.std(image)
        image = (image - mean) / (std + 1e-8)
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