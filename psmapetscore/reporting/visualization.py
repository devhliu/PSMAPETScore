"""
Visualization tools for PSMA-PET analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import SimpleITK as sitk


class Visualization:
    """
    Class for creating visualizations of PSMA-PET analysis.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualization generator.
        
        Args:
            output_dir: Directory where visualizations will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up color maps
        self.pet_cmap = plt.cm.hot
        self.ct_cmap = plt.cm.gray
        self.mri_cmap = plt.cm.gray
        self.segmentation_cmap = plt.cm.tab10
    
    def create_overlay(self, 
                      anatomical_image: np.ndarray, 
                      pet_image: np.ndarray, 
                      segmentation: Optional[np.ndarray] = None,
                      slice_idx: Optional[int] = None,
                      alpha: float = 0.5,
                      output_filename: Optional[str] = None) -> str:
        """
        Create an overlay of PET on anatomical image with optional segmentation.
        
        Args:
            anatomical_image: CT or MRI image (3D)
            pet_image: PET SUV image (3D)
            segmentation: Optional segmentation mask (3D)
            slice_idx: Slice index to visualize (if None, use middle slice)
            alpha: Transparency of the PET overlay
            output_filename: Filename for the output image
            
        Returns:
            Path to the generated image
        """
        # Determine slice index if not provided
        if slice_idx is None:
            slice_idx = anatomical_image.shape[0] // 2
        
        # Extract the slice
        anatomical_slice = anatomical_image[slice_idx]
        pet_slice = pet_image[slice_idx]
        
        # Normalize images for visualization
        anatomical_norm = self._normalize_image(anatomical_slice)
        pet_norm = self._normalize_image(pet_slice)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot anatomical image
        ax.imshow(anatomical_norm, cmap=self.ct_cmap)
        
        # Overlay PET image
        ax.imshow(pet_norm, cmap=self.pet_cmap, alpha=alpha * pet_norm)
        
        # Overlay segmentation if provided
        if segmentation is not None:
            seg_slice = segmentation[slice_idx]
            if np.any(seg_slice > 0):
                # Create a mask for the segmentation
                seg_mask = seg_slice > 0
                # Create a colored segmentation overlay
                seg_overlay = np.zeros((*seg_slice.shape, 4))
                for label in np.unique(seg_slice):
                    if label == 0:  # Skip background
                        continue
                    mask = seg_slice == label
                    color = self.segmentation_cmap(label % 10)
                    seg_overlay[mask] = (*color[:3], 0.7)  # RGB + alpha
                
                ax.imshow(seg_overlay)
        
        # Remove axes
        ax.axis('off')
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = f"overlay_slice_{slice_idx}.png"
        
        # Save the figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return str(output_path)
    
    def create_mip(self, 
                  pet_image: np.ndarray, 
                  output_filename: Optional[str] = None) -> str:
        """
        Create a Maximum Intensity Projection (MIP) of the PET image.
        
        Args:
            pet_image: PET SUV image (3D)
            output_filename: Filename for the output image
            
        Returns:
            Path to the generated MIP image
        """
        # Create MIP along each axis
        mip_coronal = np.max(pet_image, axis=0)
        mip_sagittal = np.max(pet_image, axis=1)
        mip_axial = np.max(pet_image, axis=2)
        
        # Normalize for visualization
        mip_coronal_norm = self._normalize_image(mip_coronal)
        mip_sagittal_norm = self._normalize_image(mip_sagittal)
        mip_axial_norm = self._normalize_image(mip_axial)
        
        # Create figure with three views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot MIPs
        axes[0].imshow(mip_coronal_norm, cmap=self.pet_cmap)
        axes[0].set_title('Coronal MIP')
        axes[0].axis('off')
        
        axes[1].imshow(mip_sagittal_norm, cmap=self.pet_cmap)
        axes[1].set_title('Sagittal MIP')
        axes[1].axis('off')
        
        axes[2].imshow(mip_axial_norm, cmap=self.pet_cmap)
        axes[2].set_title('Axial MIP')
        axes[2].axis('off')
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = "pet_mip.png"
        
        # Save the figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return str(output_path)
    
    def create_lesion_visualization(self,
                                  pet_image: np.ndarray,
                                  lesion_mask: np.ndarray,
                                  slice_indices: Optional[List[int]] = None,
                                  output_filename: Optional[str] = None) -> str:
        """
        Create a visualization of lesions in the PET image.
        
        Args:
            pet_image: PET SUV image (3D)
            lesion_mask: Lesion segmentation mask (3D)
            slice_indices: List of slice indices to visualize (if None, find slices with lesions)
            output_filename: Filename for the output image
            
        Returns:
            Path to the generated image
        """
        # Find slices with lesions if not provided
        if slice_indices is None:
            # Find slices with lesions
            lesion_slices = []
            for i in range(lesion_mask.shape[0]):
                if np.any(lesion_mask[i] > 0):
                    lesion_slices.append(i)
            
            # If no lesions found, use middle slice
            if not lesion_slices:
                slice_indices = [pet_image.shape[0] // 2]
            else:
                # Take up to 4 slices with lesions
                if len(lesion_slices) > 4:
                    # Evenly sample from the lesion slices
                    step = len(lesion_slices) // 4
                    slice_indices = lesion_slices[::step][:4]
                else:
                    slice_indices = lesion_slices
        
        # Create a grid of images
        n_slices = len(slice_indices)
        n_cols = min(n_slices, 4)
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        # Handle single subplot case
        if n_slices == 1:
            axes = np.array([axes])
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        for i, slice_idx in enumerate(slice_indices):
            if i >= len(axes):
                break
                
            # Extract the slice
            pet_slice = pet_image[slice_idx]
            lesion_slice = lesion_mask[slice_idx]
            
            # Normalize PET image for visualization
            pet_norm = self._normalize_image(pet_slice)
            
            # Plot PET image
            axes[i].imshow(pet_norm, cmap=self.pet_cmap)
            
            # Overlay lesion contours
            if np.any(lesion_slice > 0):
                for label in np.unique(lesion_slice):
                    if label == 0:  # Skip background
                        continue
                    
                    # Create binary mask for this lesion
                    mask = lesion_slice == label
                    
                    # Plot contour
                    from skimage import measure
                    contours = measure.find_contours(mask, 0.5)
                    for contour in contours:
                        axes[i].plot(contour[:, 1], contour[:, 0], 'c-', linewidth=2)
            
            axes[i].set_title(f'Slice {slice_idx}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n_slices, len(axes)):
            axes[i].axis('off')
            axes[i].set_visible(False)
        
        # Generate output filename if not provided
        if output_filename is None:
            output_filename = "lesion_visualization.png"
        
        # Save the figure
        output_path = self.output_dir / output_filename
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return str(output_path)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for visualization.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        # Handle empty or constant images
        if np.min(image) == np.max(image):
            return np.zeros_like(image)
        
        # Normalize to [0, 1]
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        return normalized