"""
Base class for nnUNetv2-based segmentation.
"""

import os
import torch
import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
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
        
        # Initialize nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=self.verbose,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        # Initialize the model
        self.predictor.initialize_from_trained_model_folder(
            str(self.model_folder),
            use_folds=(0,),  # Default to using fold 0
            checkpoint_name='checkpoint_final.pth',
        )
    
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
        
        # Create temporary input and output folders
        input_folder = self.model_folder / "temp_input"
        output_folder = self.model_folder / "predictions"
        input_folder.mkdir(exist_ok=True, parents=True)
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Save preprocessed images to input folder and prepare file paths
        input_files = []
        output_files = []
        
        for i, img in enumerate(preprocessed):
            # Create a unique filename for this image
            input_filename = f"image_{i:03d}.nii.gz"
            output_filename = f"pred_{i:03d}.nii.gz"
            
            # Save as NIfTI file
            img_sitk = sitk.GetImageFromArray(img)
            sitk.WriteImage(img_sitk, str(input_folder / input_filename))
            
            # Add to file lists (nnUNetPredictor expects nested lists for input files)
            input_files.append([str(input_folder / input_filename)])
            output_files.append(str(output_folder / output_filename))
        
        # Run prediction using nnUNetPredictor
        predictions = self.predictor.predict_from_files(
            input_files,
            output_files,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2
        )
        
        # If predictions were returned directly, use them
        # Otherwise, load from output files
        if predictions is None:
            # Load predictions from output folder
            prediction_files = sorted(list(output_folder.glob("*.nii.gz")))
            predictions = [sitk.GetArrayFromImage(sitk.ReadImage(str(f))) for f in prediction_files]
        
        # Postprocess
        result = self.postprocess(predictions[0] if len(predictions) == 1 else predictions)
        
        metadata = {
            "model_name": self.task_name,
            "model_folder": str(self.model_folder),
            "device": str(self.device)
        }
        
        return result, metadata