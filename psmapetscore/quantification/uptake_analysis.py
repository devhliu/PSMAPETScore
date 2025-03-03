"""
Analysis of PSMA-PET uptake in different anatomical regions.
"""

import numpy as np
import SimpleITK as sitk
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class UptakeAnalysis:
    """
    Class for analyzing PSMA-PET uptake in different anatomical regions.
    """
    
    def __init__(self):
        """
        Initialize the uptake analyzer.
        """
        # Define reference organs for background normalization
        self.reference_organs = {
            "liver": 5,
            "blood_pool": 9,  # Assuming blood pool is labeled as 9
            "parotid": 10,    # Assuming parotid gland is labeled as 10
            "spleen": 11      # Assuming spleen is labeled as 11
        }
    
    def calculate_suv_metrics(self, 
                            pet_image: np.ndarray, 
                            mask: np.ndarray, 
                            label: int = 1) -> Dict:
        """
        Calculate SUV metrics for a specific region.
        
        Args:
            pet_image: PET SUV image
            mask: Segmentation mask
            label: Label value in the mask to analyze
            
        Returns:
            Dictionary with SUV metrics
        """
        # Extract region from mask
        region_mask = (mask == label)
        
        # Extract SUV values within the region
        region_suvs = pet_image[region_mask]
        
        if len(region_suvs) == 0:
            return {
                "SUVmax": 0,
                "SUVmean": 0,
                "SUVmedian": 0,
                "SUVstd": 0,
                "SUVpeak": 0,
                "volume_ml": 0,
                "TLU": 0  # Total Lesion Uptake (SUVmean × Volume)
            }
        
        # Calculate metrics
        suv_max = np.max(region_suvs)
        suv_mean = np.mean(region_suvs)
        suv_median = np.median(region_suvs)
        suv_std = np.std(region_suvs)
        
        # Calculate SUVpeak (highest average in a 1cm³ sphere)
        # This is a simplified version - a proper implementation would use a 3D sphere kernel
        if len(region_suvs) >= 27:  # Approximate 1cm³ with 3x3x3 voxels
            sorted_suvs = np.sort(region_suvs)
            suv_peak = np.mean(sorted_suvs[-27:])
        else:
            suv_peak = suv_mean
        
        # Calculate volume in ml (assuming voxel size is in mm)
        # In a real implementation, this would use the actual voxel dimensions
        volume_ml = np.sum(region_mask) * 0.001  # Assuming 1x1x1 mm voxels
        
        # Calculate Total Lesion Uptake
        tlu = suv_mean * volume_ml
        
        return {
            "SUVmax": suv_max,
            "SUVmean": suv_mean,
            "SUVmedian": suv_median,
            "SUVstd": suv_std,
            "SUVpeak": suv_peak,
            "volume_ml": volume_ml,
            "TLU": tlu
        }
    
    def calculate_tumor_to_background_ratio(self, 
                                          tumor_metrics: Dict, 
                                          background_metrics: Dict) -> Dict:
        """
        Calculate tumor-to-background ratios.
        
        Args:
            tumor_metrics: SUV metrics for the tumor
            background_metrics: SUV metrics for the background
            
        Returns:
            Dictionary with tumor-to-background ratios
        """
        # Avoid division by zero
        if background_metrics["SUVmean"] == 0:
            return {
                "TBRmax": 0,
                "TBRmean": 0,
                "TBRpeak": 0
            }
        
        # Calculate ratios
        tbr_max = tumor_metrics["SUVmax"] / background_metrics["SUVmean"]
        tbr_mean = tumor_metrics["SUVmean"] / background_metrics["SUVmean"]
        tbr_peak = tumor_metrics["SUVpeak"] / background_metrics["SUVmean"]
        
        return {
            "TBRmax": tbr_max,
            "TBRmean": tbr_mean,
            "TBRpeak": tbr_peak
        }
    
    def analyze_lesions(self, 
                      pet_image: np.ndarray, 
                      lesion_mask: np.ndarray, 
                      anatomical_mask: np.ndarray) -> Dict:
        """
        Analyze PSMA uptake in lesions.
        
        Args:
            pet_image: PET SUV image
            lesion_mask: Mask with labeled lesions (each lesion has a unique label)
            anatomical_mask: Segmentation mask with anatomical regions
            
        Returns:
            Dictionary with lesion analysis
        """
        # Get unique lesion labels
        lesion_labels = np.unique(lesion_mask)
        lesion_labels = lesion_labels[lesion_labels > 0]  # Exclude background
        
        # Analyze each lesion
        lesions = []
        for label in lesion_labels:
            # Create binary mask for this lesion
            lesion_binary = (lesion_mask == label)
            
            # Calculate SUV metrics
            metrics = self.calculate_suv_metrics(pet_image, lesion_mask, label)
            
            # Determine anatomical region
            region_i