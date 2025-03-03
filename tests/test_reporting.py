import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from psmapetscore.reporting import ReportGenerator, Visualization

def test_report_generation():
    """
    Test the report generation functionality with sample data.
    """
    # Create output directories
    output_dir = Path(os.path.join(Path(__file__).parent, "output"))
    vis_output_dir = Path(os.path.join(output_dir, "visualizations"))
    report_output_dir = Path(os.path.join(output_dir, "reports"))

    
    # Create directories if they don't exist
    output_dir.mkdir(exist_ok=True)
    vis_output_dir.mkdir(exist_ok=True)
    report_output_dir.mkdir(exist_ok=True)
    
    # Create sample data
    # 1. Create sample images (3D arrays)
    image_shape = (50, 100, 100)  # z, y, x
    
    # Create a sample PET image with a hot spot
    pet_image = np.zeros(image_shape)
    # Add a "lesion" in the center
    center_z, center_y, center_x = image_shape[0]//2, image_shape[1]//2, image_shape[2]//2
    for z in range(center_z-5, center_z+5):
        for y in range(center_y-10, center_y+10):
            for x in range(center_x-10, center_x+10):
                # Create a spherical-like intensity pattern
                dist = np.sqrt((z-center_z)**2 + (y-center_y)**2 + (x-center_x)**2)
                if dist < 10:
                    pet_image[z, y, x] = max(0, 10 - dist) * 2  # Higher in center
    
    # Create a sample CT image
    ct_image = np.ones(image_shape) * 100  # Background
    # Add some structure
    for z in range(image_shape[0]):
        ct_image[z, 30:70, 30:70] = 200  # "Body"
    
    # Create a sample segmentation mask
    segmentation = np.zeros(image_shape, dtype=np.int32)
    # Add a segmentation for the lesion
    for z in range(center_z-3, center_z+3):
        for y in range(center_y-7, center_y+7):
            for x in range(center_x-7, center_x+7):
                dist = np.sqrt((z-center_z)**2 + (y-center_y)**2 + (x-center_x)**2)
                if dist < 7:
                    segmentation[z, y, x] = 1  # Label 1 for the lesion
    
    # 2. Create sample patient information
    patient_info = {
        "PatientID": "TEST12345",
        "PatientName": "Test Patient",
        "StudyDate": datetime.now().strftime("%Y%m%d"),
        "Modality": "PT/CT",
        "Radiopharmaceutical": "68Ga-PSMA-11"
    }
    
    # 3. Create sample primary lesion evaluation
    primary_evaluation = {
        "region_name": "Prostate",
        "SUVmax": 15.7,
        "SUVmean": 8.3,
        "SUVpeak": 12.4,
        "volume_ml": 4.2,
        "miPSMA_score": "3"
    }
    
    # 4. Create sample whole-body evaluation
    whole_body_evaluation = {
        "total_lesion_count": 3,
        "total_lesion_volume_ml": 7.8,
        "max_SUVmax": 15.7,
        "regions": {
            "1": {
                "name": "Prostate",
                "lesion_count": 1,
                "max_SUVmax": 15.7,
                "total_volume_ml": 4.2,
                "lesions": [
                    {
                        "lesion_id": 1,
                        "SUVmax": 15.7,
                        "SUVmean": 8.3,
                        "volume_ml": 4.2,
                        "miPSMA_score": "3"
                    }
                ]
            },
            "2": {
                "name": "Lymph Nodes",
                "lesion_count": 2,
                "max_SUVmax": 10.2,
                "total_volume_ml": 3.6,
                "lesions": [
                    {
                        "lesion_id": 2,
                        "SUVmax": 10.2,
                        "SUVmean": 6.1,
                        "volume_ml": 2.1,
                        "miPSMA_score": "2"
                    },
                    {
                        "lesion_id": 3,
                        "SUVmax": 8.5,
                        "SUVmean": 5.2,
                        "volume_ml": 1.5,
                        "miPSMA_score": "2"
                    }
                ]
            }
        }
    }
    
    # Create visualizations
    visualizer = Visualization(str(vis_output_dir))
    
    # Create overlay visualization
    overlay_path = visualizer.create_overlay(
        anatomical_image=ct_image,
        pet_image=pet_image,
        segmentation=segmentation,
        output_filename="test_overlay.png"
    )
    print(f"Created overlay visualization: {overlay_path}")
    
    # Create MIP visualization
    mip_path = visualizer.create_mip(
        pet_image=pet_image,
        output_filename="test_mip.png"
    )
    print(f"Created MIP visualization: {mip_path}")
    
    # Create lesion visualization
    lesion_vis_path = visualizer.create_lesion_visualization(
        pet_image=pet_image,
        lesion_mask=segmentation,
        output_filename="test_lesion_vis.png"
    )
    print(f"Created lesion visualization: {lesion_vis_path}")
    
    # Collect visualization data for the report
    visualization_data = {
        "primary_lesion_image": overlay_path,
        "whole_body_mip": mip_path
    }
    
    # Generate the report
    report_generator = ReportGenerator(str(report_output_dir))
    report_path = report_generator.generate_report(
        patient_info=patient_info,
        primary_evaluation=primary_evaluation,
        whole_body_evaluation=whole_body_evaluation,
        visualization_data=visualization_data,
        output_filename="test_report.pdf"
    )
    
    print(f"Generated report: {report_path}")
    
    # Note: DICOM SR generation requires a reference DICOM file
    # This part is commented out as it requires an actual DICOM file
    """
    # Generate DICOM SR (if a reference DICOM file is available)
    reference_dicom_path = "path/to/reference.dcm"
    if os.path.exists(reference_dicom_path):
        dicom_sr_path = report_generator.generate_dicom_sr(
            patient_info=patient_info,
            primary_evaluation=primary_evaluation,
            whole_body_evaluation=whole_body_evaluation,
            reference_dicom_path=reference_dicom_path,
            output_filename="test_dicom_sr.dcm"
        )
        print(f"Generated DICOM SR: {dicom_sr_path}")
    """

if __name__ == "__main__":
    test_report_generation()