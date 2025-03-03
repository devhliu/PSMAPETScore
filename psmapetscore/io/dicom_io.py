"""
DICOM I/O handling for PSMA-PET and CT/MRI images.
"""

import os
import pydicom
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional


class DicomReader:
    """
    Class for reading DICOM series from different modalities.
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the DICOM reader with the base directory containing modality folders.
        
        Args:
            base_dir: Base directory containing modality-specific folders
        """
        self.base_dir = Path(base_dir)
        self.modalities = {}
        self.series_info = {}
        
    def scan_directories(self) -> Dict[str, List[str]]:
        """
        Scan the base directory for modality folders and DICOM series.
        
        Returns:
            Dictionary mapping modality names to series paths
        """
        modality_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        
        for modality_dir in modality_dirs:
            modality_name = modality_dir.name
            series_dirs = [d for d in modality_dir.iterdir() if d.is_dir()]
            
            self.modalities[modality_name] = [str(s) for s in series_dirs]
            
        return self.modalities
    
    def read_series(self, series_path: str) -> Tuple[sitk.Image, Dict]:
        """
        Read a DICOM series and return as SimpleITK image with metadata.
        
        Args:
            series_path: Path to the DICOM series folder
            
        Returns:
            SimpleITK image and metadata dictionary
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(series_path)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # Extract metadata from first slice
        first_slice = pydicom.dcmread(dicom_names[0])
        metadata = {
            'PatientID': first_slice.PatientID,
            'PatientName': str(first_slice.PatientName),
            'Modality': first_slice.Modality,
            'SeriesDescription': first_slice.SeriesDescription if hasattr(first_slice, 'SeriesDescription') else '',
            'StudyDate': first_slice.StudyDate,
            'SeriesNumber': first_slice.SeriesNumber,
            'SliceThickness': first_slice.SliceThickness if hasattr(first_slice, 'SliceThickness') else 0,
        }
        
        # Add PET-specific metadata if available
        if first_slice.Modality == 'PT':
            if hasattr(first_slice, 'RadiopharmaceuticalInformationSequence'):
                radiopharm_seq = first_slice.RadiopharmaceuticalInformationSequence[0]
                if hasattr(radiopharm_seq, 'RadionuclideTotalDose'):
                    metadata['RadionuclideTotalDose'] = radiopharm_seq.RadionuclideTotalDose
                if hasattr(radiopharm_seq, 'RadionuclideHalfLife'):
                    metadata['RadionuclideHalfLife'] = radiopharm_seq.RadionuclideHalfLife
                if hasattr(radiopharm_seq, 'RadiopharmaceuticalStartTime'):
                    metadata['RadiopharmaceuticalStartTime'] = radiopharm_seq.RadiopharmaceuticalStartTime
            
            # Check for PSMA tracer information
            if hasattr(first_slice, 'RadiopharmaceuticalInformationSequence'):
                radiopharm_seq = first_slice.RadiopharmaceuticalInformationSequence[0]
                if hasattr(radiopharm_seq, 'RadiopharmaceuticalStartTime'):
                    if hasattr(radiopharm_seq, 'RadionuclideCodeSequence'):
                        code_seq = radiopharm_seq.RadionuclideCodeSequence[0]
                        if hasattr(code_seq, 'CodeMeaning'):
                            metadata['Radionuclide'] = code_seq.CodeMeaning
        
        self.series_info[series_path] = metadata
        return image, metadata
    
    def read_modality(self, modality: str) -> Dict[str, Tuple[sitk.Image, Dict]]:
        """
        Read all series for a specific modality.
        
        Args:
            modality: Modality name (e.g., 'CT', 'PT', 'MR')
            
        Returns:
            Dictionary mapping series paths to (image, metadata) tuples
        """
        if modality not in self.modalities:
            self.scan_directories()
            
        if modality not in self.modalities:
            raise ValueError(f"Modality {modality} not found in {self.base_dir}")
            
        result = {}
        for series_path in self.modalities[modality]:
            result[series_path] = self.read_series(series_path)
            
        return result


class DicomWriter:
    """
    Class for writing segmentation results and reports back to DICOM format.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the DICOM writer with the output directory.
        
        Args:
            output_dir: Directory where DICOM files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def write_segmentation(self, 
                          segmentation: np.ndarray, 
                          reference_series: str,
                          output_series_description: str,
                          series_number: int = 1000,
                          label_map: Optional[Dict[int, str]] = None) -> str:
        """
        Write a segmentation mask as a DICOM Segmentation object.
        
        Args:
            segmentation: 3D numpy array containing segmentation labels
            reference_series: Path to the reference DICOM series
            output_series_description: Description for the output series
            series_number: Series number for the output
            label_map: Dictionary mapping label values to anatomical names
            
        Returns:
            Path to the output DICOM segmentation folder
        """
        # Read reference series to get metadata
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(reference_series)
        reader.SetFileNames(dicom_names)
        reference_image = reader.Execute()
        
        # Convert segmentation to SimpleITK image with same geometry
        seg_image = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        seg_image.SetOrigin(reference_image.GetOrigin())
        seg_image.SetSpacing(reference_image.GetSpacing())
        seg_image.SetDirection(reference_image.GetDirection())
        
        # Create output directory
        first_slice = pydicom.dcmread(dicom_names[0])
        patient_id = first_slice.PatientID
        output_subdir = self.output_dir / f"{patient_id}_SEG_{series_number}"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Write segmentation as DICOM-SEG
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("GDCMImageIO")
        
        # Get metadata from reference series
        reference_dcm = pydicom.dcmread(dicom_names[0])
        
        # Create a new DICOM series
        for i in range(segmentation.shape[0]):
            # Extract the slice
            slice_image = seg_image[:, :, i]
            
            # Create a new DICOM file for this slice
            dcm = pydicom.Dataset()
            
            # Copy essential DICOM attributes from reference
            dcm.PatientID = reference_dcm.PatientID
            dcm.PatientName = reference_dcm.PatientName
            dcm.StudyInstanceUID = reference_dcm.StudyInstanceUID
            
            # Generate new UIDs for series and instance
            dcm.SeriesInstanceUID = pydicom.uid.generate_uid()
            dcm.SOPInstanceUID = pydicom.uid.generate_uid()
            
            # Set segmentation-specific attributes
            dcm.Modality = "SEG"
            dcm.SeriesDescription = output_series_description
            dcm.SeriesNumber = series_number
            
            # Set the pixel data
            dcm.PixelData = slice_image.tobytes()
            
            # Save the DICOM file
            output_file = output_subdir / f"slice_{i:04d}.dcm"
            dcm.save_as(output_file)
        
        return str(output_subdir)
    
    def write_structured_report(self,
                               report_data: Dict,
                               reference_series: str,
                               output_filename: str) -> str:
        """
        Write a structured report as DICOM-SR.
        
        Args:
            report_data: Dictionary containing report data
            reference_series: Path to the reference DICOM series
            output_filename: Filename for the output DICOM-SR
            
        Returns:
            Path to the output DICOM-SR file
        """
        # This is a placeholder for DICOM-SR creation
        # In a real implementation, this would create a proper DICOM-SR document
        
        # Read reference series to get patient metadata
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(reference_series)
        reference_dcm = pydicom.dcmread(dicom_names[0])
        
        # Create a basic SR document
        sr = pydicom.Dataset()
        
        # Copy patient and study information
        sr.PatientID = reference_dcm.PatientID
        sr.PatientName = reference_dcm.PatientName
        sr.StudyInstanceUID = reference_dcm.StudyInstanceUID
        
        # Generate new UIDs
        sr.SeriesInstanceUID = pydicom.uid.generate_uid()
        sr.SOPInstanceUID = pydicom.uid.generate_uid()
        
        # Set SR-specific attributes
        sr.Modality = "SR"
        sr.SeriesDescription = "PSMA-PET Structured Report"
        
        # Save the SR document
        output_path = self.output_dir / output_filename
        sr.save_as(output_path)
        
        return str(output_path)