"""
Generation of structured reports for PSMA-PET analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import pydicom
import SimpleITK as sitk
from pylatex import Document, Section, Subsection, Tabular, Figure, Command
from pylatex.utils import bold, italic


class ReportGenerator:
    """
    Class for generating structured reports from PSMA-PET analysis.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory where reports will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self, 
                       patient_info: Dict, 
                       primary_evaluation: Dict, 
                       whole_body_evaluation: Dict,
                       visualization_data: Dict,
                       output_filename: Optional[str] = None) -> str:
        """
        Generate a structured report for PSMA-PET analysis.
        
        Args:
            patient_info: Patient information
            primary_evaluation: Primary (prostate) lesion evaluation
            whole_body_evaluation: Whole-body lesion evaluation
            visualization_data: Visualization data (images, etc.)
            output_filename: Filename for the output report
            
        Returns:
            Path to the generated report
        """
        # Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patient_id = patient_info.get("PatientID", "unknown")
            output_filename = f"{patient_id}_PSMA_Report_{timestamp}.pdf"
        
        # Create LaTeX document
        doc = Document(output_filename.replace(".pdf", ""))
        
        # Add preamble
        doc.preamble.append(Command('title', 'PSMA-PET Structured Report'))
        doc.preamble.append(Command('author', f'Patient ID: {patient_info.get("PatientID", "Unknown")}'))
        doc.preamble.append(Command('date', datetime.now().strftime("%Y-%m-%d")))
        
        # Create document
        doc.append(Command('maketitle'))
        
        # Add patient information section
        with doc.create(Section('Patient Information')):
            with doc.create(Tabular('|l|l|')) as table:
                table.add_hline()
                table.add_row(["Patient ID", patient_info.get("PatientID", "Unknown")])
                table.add_hline()
                table.add_row(["Patient Name", patient_info.get("PatientName", "Unknown")])
                table.add_hline()
                table.add_row(["Study Date", patient_info.get("StudyDate", "Unknown")])
                table.add_hline()
                table.add_row(["Modality", patient_info.get("Modality", "Unknown")])
                table.add_hline()
                table.add_row(["Radiopharmaceutical", patient_info.get("Radiopharmaceutical", "Unknown")])
                table.add_hline()
        
        # Add primary lesion section
        with doc.create(Section('Primary Lesion Evaluation')):
            with doc.create(Tabular('|l|l|')) as table:
                table.add_hline()
                table.add_row(["Region", primary_evaluation.get("region_name", "Unknown")])
                table.add_hline()
                table.add_row(["SUVmax", f"{primary_evaluation.get('SUVmax', 0):.2f}"])
                table.add_hline()
                table.add_row(["SUVmean", f"{primary_evaluation.get('SUVmean', 0):.2f}"])
                table.add_hline()
                table.add_row(["SUVpeak", f"{primary_evaluation.get('SUVpeak', 0):.2f}"])
                table.add_hline()
                table.add_row(["Volume (ml)", f"{primary_evaluation.get('volume_ml', 0):.2f}"])
                table.add_hline()
                table.add_row(["miPSMA Score", primary_evaluation.get("miPSMA_score", "Unknown")])
                table.add_hline()
            
            # Add primary lesion visualization if available
            if "primary_lesion_image" in visualization_data:
                with doc.create(Figure(position='h!')) as fig:
                    fig.add_image(visualization_data["primary_lesion_image"], width='8cm')
                    fig.add_caption('Primary Lesion Visualization')
        
        # Add whole-body evaluation section
        with doc.create(Section('Whole-Body Evaluation')):
            doc.append(f"Total Lesion Count: {whole_body_evaluation.get('total_lesion_count', 0)}")
            doc.append(f"\nTotal Lesion Volume: {whole_body_evaluation.get('total_lesion_volume_ml', 0):.2f} ml")
            doc.append(f"\nMaximum SUVmax: {whole_body_evaluation.get('max_SUVmax', 0):.2f}")
            
            # Add region-specific subsections
            regions = whole_body_evaluation.get("regions", {})
            for region_id, region_data in regions.items():
                with doc.create(Subsection(region_data["name"])):
                    doc.append(f"Lesion Count: {region_data['lesion_count']}")
                    doc.append(f"\nMaximum SUVmax: {region_data['max_SUVmax']:.2f}")
                    doc.append(f"\nTotal Volume: {region_data['total_volume_ml']:.2f} ml")
                    
                    # Add table of lesions in this region
                    if len(region_data["lesions"]) > 0:
                        with doc.create(Tabular('|l|l|l|l|l|')) as table:
                            table.add_hline()
                            table.add_row(["Lesion ID", "SUVmax", "SUVmean", "Volume (ml)", "miPSMA Score"])
                            table.add_hline()
                            for lesion in region_data["lesions"]:
                                table.add_row([
                                    str(lesion["lesion_id"]),
                                    f"{lesion['SUVmax']:.2f}",
                                    f"{lesion['SUVmean']:.2f}",
                                    f"{lesion['volume_ml']:.2f}",
                                    lesion["miPSMA_score"]
                                ])
                                table.add_hline()
            
            # Add whole-body visualization if available
            if "whole_body_mip" in visualization_data:
                with doc.create(Figure(position='h!')) as fig:
                    fig.add_image(visualization_data["whole_body_mip"], width='10cm')
                    fig.add_caption('Whole-Body Maximum Intensity Projection')
        
        # Generate the PDF
        output_path = self.output_dir / output_filename
        doc.generate_pdf(str(output_path).replace(".pdf", ""), clean_tex=True)
        
        return str(output_path)
    
    def generate_dicom_sr(self, 
                         patient_info: Dict, 
                         primary_evaluation: Dict, 
                         whole_body_evaluation: Dict,
                         reference_dicom_path: str,
                         output_filename: Optional[str] = None) -> str:
        """
        Generate a DICOM Structured Report for PSMA-PET analysis.
        
        Args:
            patient_info: Patient information
            primary_evaluation: Primary (prostate) lesion evaluation
            whole_body_evaluation: Whole-body lesion evaluation
            reference_dicom_path: Path to a reference DICOM file
            output_filename: Filename for the output DICOM SR
            
        Returns:
            Path to the generated DICOM SR
        """
        # Generate output filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            patient_id = patient_info.get("PatientID", "unknown")
            output_filename = f"{patient_id}_PSMA_SR_{timestamp}.dcm"
        
        # Read reference DICOM file
        reference_dcm = pydicom.dcmread(reference_dicom_path)
        
        # Create a new DICOM SR object
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
        sr.ContentDate = datetime.now().strftime("%Y%m%d")
        sr.ContentTime = datetime.now().strftime("%H%M%S")
        
        # Save the SR document
        output_path = self.output_dir / output_filename
        sr.save_as(output_path)
        
        return str(output_path)