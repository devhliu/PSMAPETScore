"""
Implementation of PROMISE v2 criteria for PSMA-PET quantification.
"""

import numpy as np
import SimpleITK as sitk
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PROMISECriteria:
    """
    Class for implementing PROMISE v2 criteria for PSMA-PET quantification.
    """
    
    def __init__(self):
        """
        Initialize the PROMISE criteria evaluator.
        """
        # Define anatomical regions according to PROMISE v2
        self.anatomical_regions = {
            1: "Prostate/Prostate Bed",
            2: "Pelvic Lymph Nodes",
            3: "Retroperitoneal Lymph Nodes",
            4: "Supradiaphragmatic Lymph Nodes",
            5: "Liver",
            6: "Lungs",
            7: "Bones",
            8: "Other Visceral Organs"
        }
        
        # Define PROMISE v2 scoring thresholds
        self.suv_thresholds = {
            "miPSMA-0": 0,
            "miPSMA-1": 1.0,
            "miPSMA-2": 2.0,
            "miPSMA-3": 3.0
        }
    
    def classify_lesion(self, 
                       lesion_mask: np.ndarray, 
                       pet_image: np.ndarray, 
                       anatomical_mask: np.ndarray) -> Dict:
        """
        Classify a lesion according to PROMISE v2 criteria.
        
        Args:
            lesion_mask: Binary mask of the lesion
            pet_image: PET SUV image
            anatomical_mask: Segmentation mask with anatomical regions
            
        Returns:
            Dictionary with lesion classification
        """
        # Get the anatomical region of the lesion
        lesion_region = self._get_lesion_region(lesion_mask, anatomical_mask)
        
        # Calculate SUV metrics for the lesion
        suv_metrics = self._calculate_suv_metrics(lesion_mask, pet_image)
        
        # Determine miPSMA score
        mipsma_score = self._determine_mipsma_score(suv_metrics["SUVmax"])
        
        # Create classification result
        classification = {
            "region": lesion_region,
            "region_name": self.anatomical_regions.get(lesion_region, "Unknown"),
            "SUVmax": suv_metrics["SUVmax"],
            "SUVmean": suv_metrics["SUVmean"],
            "SUVpeak": suv_metrics["SUVpeak"],
            "volume_ml": suv_metrics["volume_ml"],
            "miPSMA_score": mipsma_score
        }
        
        return classification
    
    def _get_lesion_region(self, lesion_mask: np.ndarray, anatomical_mask: np.ndarray) -> int:
        """
        Determine the anatomical region of a lesion.
        
        Args:
            lesion_mask: Binary mask of the lesion
            anatomical_mask: Segmentation mask with anatomical regions
            
        Returns:
            Region ID according to PROMISE v2
        """
        # Find overlap between lesion and anatomical regions
        overlap = lesion_mask * anatomical_mask
        
        # Count voxels in each region
        unique, counts = np.unique(overlap[overlap > 0], return_counts=True)
        
        if len(unique) == 0:
            return 0  # Unknown region
        
        # Return the region with the most overlap
        return unique[np.argmax(counts)]
    
    def _calculate_suv_metrics(self, lesion_mask: np.ndarray, pet_image: np.ndarray) -> Dict:
        """
        Calculate SUV metrics for a lesion.
        
        Args:
            lesion_mask: Binary mask of the lesion
            pet_image: PET SUV image
            
        Returns:
            Dictionary with SUV metrics
        """
        # Extract SUV values within the lesion
        lesion_suvs = pet_image[lesion_mask > 0]
        
        if len(lesion_suvs) == 0:
            return {
                "SUVmax": 0,
                "SUVmean": 0,
                "SUVpeak": 0,
                "volume_ml": 0
            }
        
        # Calculate metrics
        suv_max = np.max(lesion_suvs)
        suv_mean = np.mean(lesion_suvs)
        
        # Calculate SUVpeak (highest average in a 1cm³ sphere)
        # This is a simplified version - a proper implementation would use a 3D sphere kernel
        if len(lesion_suvs) >= 27:  # Approximate 1cm³ with 3x3x3 voxels
            sorted_suvs = np.sort(lesion_suvs)
            suv_peak = np.mean(sorted_suvs[-27:])
        else:
            suv_peak = suv_mean
        
        # Calculate volume in ml (assuming voxel size is in mm)
        # In a real implementation, this would use the actual voxel dimensions
        volume_ml = np.sum(lesion_mask) * 0.001  # Assuming 1x1x1 mm voxels
        
        return {
            "SUVmax": suv_max,
            "SUVmean": suv_mean,
            "SUVpeak": suv_peak,
            "volume_ml": volume_ml
        }
    
    def _determine_mipsma_score(self, suv_max: float) -> str:
        """
        Determine miPSMA score based on SUVmax.
        
        Args:
            suv_max: Maximum SUV value
            
        Returns:
            miPSMA score string
        """
        if suv_max < self.suv_thresholds["miPSMA-1"]:
            return "miPSMA-0"
        elif suv_max < self.suv_thresholds["miPSMA-2"]:
            return "miPSMA-1"
        elif suv_max < self.suv_thresholds["miPSMA-3"]:
            return "miPSMA-2"
        else:
            return "miPSMA-3"
    
    def evaluate_primary_lesion(self, 
                              prostate_mask: np.ndarray, 
                              pet_image: np.ndarray) -> Dict:
        """
        Evaluate primary (prostate) lesion.
        
        Args:
            prostate_mask: Binary mask of the prostate
            pet_image: PET SUV image
            
        Returns:
            Dictionary with primary lesion evaluation
        """
        # Calculate SUV metrics for the prostate
        suv_metrics = self._calculate_suv_metrics(prostate_mask, pet_image)
        
        # Determine miPSMA score
        mipsma_score = self._determine_mipsma_score(suv_metrics["SUVmax"])
        
        # Create evaluation result
        evaluation = {
            "region": 1,  # Prostate/Prostate Bed
            "region_name": self.anatomical_regions[1],
            "SUVmax": suv_metrics["SUVmax"],
            "SUVmean": suv_metrics["SUVmean"],
            "SUVpeak": suv_metrics["SUVpeak"],
            "volume_ml": suv_metrics["volume_ml"],
            "miPSMA_score": mipsma_score
        }
        
        return evaluation
    
    def evaluate_whole_body(self, 
                          lesion_masks: List[np.ndarray], 
                          pet_image: np.ndarray, 
                          anatomical_mask: np.ndarray) -> Dict:
        """
        Evaluate whole-body lesions according to PROMISE v2.
        
        Args:
            lesion_masks: List of binary masks for each lesion
            pet_image: PET SUV image
            anatomical_mask: Segmentation mask with anatomical regions
            
        Returns:
            Dictionary with whole-body evaluation
        """
        # Classify each lesion
        lesions = []
        for i, mask in enumerate(lesion_masks):
            classification = self.classify_lesion(mask, pet_image, anatomical_mask)
            classification["lesion_id"] = i + 1
            lesions.append(classification)
        
        # Group lesions by anatomical region
        regions = {}
        for region_id in self.anatomical_regions:
            region_lesions = [l for l in lesions if l["region"] == region_id]
            if region_lesions:
                regions[region_id] = {
                    "name": self.anatomical_regions[region_id],
                    "lesion_count": len(region_lesions),
                    "max_SUVmax": max([l["SUVmax"] for l in region_lesions]),
                    "total_volume_ml": sum([l["volume_ml"] for l in region_lesions]),
                    "lesions": region_lesions
                }
        
        # Calculate overall metrics
        total_lesion_count = len(lesions)
        total_lesion_volume = sum([l["volume_ml"] for l in lesions])
        max_suv = max([l["SUVmax"] for l in lesions]) if lesions else 0
        
        # Create whole-body evaluation
        evaluation = {
            "total_lesion_count": total_lesion_count,
            "total_lesion_volume_ml": total_lesion_volume,
            "max_SUVmax": max_suv,
            "regions": regions,
            "lesions": lesions
        }
        
        return evaluation