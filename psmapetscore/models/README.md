# Pre-trained Models for PSMAPETScore

This directory contains pre-trained models for the PSMAPETScore package.

## Directory Structure

- `ct_segmentation/`: Contains nnUNetv2 models for CT whole-body segmentation
- `mri_segmentation/`: Contains nnUNetv2 models for MRI whole-body segmentation
- `pet_lesion/`: Contains nnUNetv2 models for PSMA-PET lesion segmentation

## Adding Models

To add a pre-trained model:

1. Create a subdirectory in the appropriate folder (e.g., `ct_segmentation/`)
2. Copy the nnUNetv2 model files into the subdirectory
3. The model will be automatically detected and used by the corresponding segmentation class

## Model Sources

The models should be trained using nnUNetv2 with the following tasks:

- CT Segmentation: Task503_CTWholebody
- MRI Segmentation: Task504_MRIWholebody
- PET Lesion Segmentation: Task505_PSMALesion

For more information on training nnUNetv2 models, see the [nnUNetv2 documentation](https://github.com/MIC-DKFZ/nnUNet).