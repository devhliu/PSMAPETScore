"""
NIfTI I/O handling for internal processing.
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, Tuple, Optional


class NiftiConverter:
    """
    Class for converting between DICOM and NIfTI formats.
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize the NIfTI converter.
        
        Args:
            temp_dir: Directory for temporary NIfTI files
        """
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp"))
            self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def dicom_to_nifti(self, dicom_series_path: str, output_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Convert a DICOM series to NIfTI format.
        
        Args:
            dicom_series_path: Path to the DICOM series folder
            output_path: Path where the NIfTI file will be saved
            
        Returns:
            Path to the NIfTI file and metadata dictionary
        """
        # Read the DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Generate output path if not provided
        if output_path is None:
            series_name = Path(dicom_series_path).name
            output_path = str(self.temp_dir / f"{series_name}.nii.gz")
        
        # Write the NIfTI file
        sitk.WriteImage(image, output_path)
        
        # Extract metadata
        metadata = {
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'size': image.GetSize(),
            'pixel_type': image.GetPixelID(),
            'number_of_components_per_pixel': image.GetNumberOfComponentsPerPixel()
        }
        
        return output_path, metadata
    
    def nifti_to_dicom(self, 
                       nifti_path: str, 
                       reference_dicom_path: str, 
                       output_dir: Optional[str] = None) -> str:
        """
        Convert a NIfTI file back to DICOM format using a reference DICOM series.
        
        Args:
            nifti_path: Path to the NIfTI file
            reference_dicom_path: Path to the reference DICOM series folder
            output_dir: Directory where DICOM files will be saved
            
        Returns:
            Path to the output DICOM folder
        """
        # Read the NIfTI file
        nifti_image = sitk.ReadImage(nifti_path)
        
        # Read the reference DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(reference_dicom_path)
        reader.SetFileNames(dicom_names)
        reference_image = reader.Execute()
        
        # Ensure the NIfTI image has the same geometry as the reference
        nifti_image.SetOrigin(reference_image.GetOrigin())
        nifti_image.SetSpacing(reference_image.GetSpacing())
        nifti_image.SetDirection(reference_image.GetDirection())
        
        # Generate output directory if not provided
        if output_dir is None:
            nifti_name = Path(nifti_path).stem
            output_dir = str(self.temp_dir / f"{nifti_name}_dicom")
        
        # Create the output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Write the DICOM series
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("GDCMImageIO")
        
        # Get metadata from reference series
        reference_dcm = sitk.ReadImage(dicom_names[0])
        
        # Create a new DICOM series
        for i in range(nifti_image.GetDepth()):
            # Extract the slice
            slice_image = nifti_image[:, :, i]
            
            # Set the output filename
            output_file = os.path.join(output_dir, f"slice_{i:04d}.dcm")
            
            # Write the slice
            writer.SetFileName(output_file)
            writer.Execute(slice_image)
        
        return output_dir
    
    def register_images(self, 
                       moving_image_path: str, 
                       fixed_image_path: str, 
                       output_path: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Register a moving image to a fixed image.
        
        Args:
            moving_image_path: Path to the moving image (NIfTI)
            fixed_image_path: Path to the fixed image (NIfTI)
            output_path: Path where the registered image will be saved
            
        Returns:
            Path to the registered image and transformation parameters
        """
        # Read the images
        moving_image = sitk.ReadImage(moving_image_path)
        fixed_image = sitk.ReadImage(fixed_image_path)
        
        # Set up the registration method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Set up similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)
        
        # Set up optimizer
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, 
                                                         numberOfIterations=100, 
                                                         convergenceMinimumValue=1e-6, 
                                                         convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        # Set up interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)
        
        # Set up initial transform
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                            moving_image, 
                                                            sitk.Euler3DTransform(), 
                                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method.SetInitialTransform(initial_transform)
        
        # Execute the registration
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        # Apply the transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(final_transform)
        
        registered_image = resampler.Execute(moving_image)
        
        # Generate output path if not provided
        if output_path is None:
            moving_name = Path(moving_image_path).stem
            fixed_name = Path(fixed_image_path).stem
            output_path = str(self.temp_dir / f"{moving_name}_to_{fixed_name}.nii.gz")
        
        # Write the registered image
        sitk.WriteImage(registered_image, output_path)
        
        # Extract transformation parameters
        transform_params = {
            'transform_type': final_transform.GetName(),
            'parameters': final_transform.GetParameters(),
            'fixed_parameters': final_transform.GetFixedParameters()
        }
        
        return output_path, transform_params