"""
Base class for nnUNetv2-based segmentation.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
from nnunetv2.paths import nnUNet_results


class BaseSegmentation:
    """
    Base class for all nnUNetv2-based segmentation models.
    """
    
    def __init__(self, 
                 model_folder: str,
                 task_name: str,
                 use_gpu: bool = True,
                 verbose: bool = False):
        """
        Initialize the segmentation model.
        
        Args:
            model_folder: Path to the trained model folder
            task_name: nnUNet task name
            use_gpu: Whether to use GPU for inference
            verbose: Whether to print detailed information
        """
        self.model_folder = Path(model_folder)
        self.task_name = task_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.verbose = verbose
        
        # Verify model exists
        if not self.model_folder.exists():
            raise FileNotFoundError(f"Model folder not found: {self.model_folder}")
            
        # Set device
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        raise NotImplementedError("Subclass must implement preprocess method")
    
    def postprocess(self, prediction: np.ndarray) -> np.ndarray:
        """
        Postprocess the model prediction.
        
        Args:
            prediction: Model prediction array
            
        Returns:
            Postprocessed prediction array
        """
        raise NotImplementedError("Subclass must implement postprocess method")
    
    def predict(self, 
               input_images: Union[np.ndarray, List[np.ndarray]], 
               input_metadata: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Run inference on input images.
        
        Args:
            input_images: Input image or list of images
            input_metadata: Optional metadata for preprocessing
            
        Returns:
            Tuple of (segmentation mask, prediction metadata)
        """
        # Ensure input is a list
        if isinstance(input_images, np.ndarray):
            input_images = [input_images]
            
        # Preprocess
        preprocessed = [self.preprocess(img) for img in input_images]
        
        # Run nnUNet prediction
        prediction = predict_from_raw_data(
            list_of_lists_or_source_folder=preprocessed,
            output_folder=self.model_folder / "predictions",
            model_folder=self.model_folder,
            use_gpu=self.use_gpu,
            verbose=self.verbose,
            save_probabilities=False,
            overwrite=True
        )
        
        # Postprocess
        result = self.postprocess(prediction)
        
        metadata = {
            "model_name": self.task_name,
            "model_folder": str(self.model_folder),
            "device": str(self.device)
        }
        
        return result, metadata