# Semantic Image Segmentation

A solo semantic segmentation project developed for a machine learning hackathon in **September 2025**.

The goal of the project was to build a multiclass image segmentation pipeline using deep learning. The solution is based on **DeepLabV3+** with an **EfficientNet-B5** encoder and was implemented using PyTorch.

## Overview

The project covers the complete semantic segmentation workflow:

- image and mask preprocessing
- data augmentation
- model configuration
- training and validation
- IoU-based model evaluation
- prediction visualization
- mask post-processing

## Model

- **Architecture:** DeepLabV3+
- **Encoder:** EfficientNet-B5
- **Encoder initialization:** ImageNet pretrained weights
- **Number of classes:** 4
- **Input resolution:** 512 × 512
- **Framework:** PyTorch

## Training

The model is trained using a combination of:

- Dice Loss
- Focal Loss

The validation metric is **Intersection over Union (IoU)**.

Training also uses:

- Adam optimizer
- ReduceLROnPlateau learning-rate scheduler
- model selection based on validation IoU

## Data Augmentation

The augmentation pipeline is implemented with Albumentations and includes:

- horizontal and vertical flips
- random 90° rotations
- shift, scale and rotation
- brightness and contrast changes
- Gaussian noise
- blur
- grid distortion
- coarse dropout

## Post-processing

Predicted segmentation masks are additionally processed using morphological operations to improve the final masks.

## Tech Stack

- Python
- PyTorch
- segmentation-models-pytorch
- Albumentations
- Rasterio
- TorchMetrics
- NumPy
- Matplotlib

## Installation

Clone the repository and install the required dependencies:

    git clone https://github.com/Mahan1341/Image-segmentation-hackathon.git
    cd Image-segmentation-hackathon
    pip install -r requirements.txt

The dataset used during the hackathon is not included in the repository.

## Repository

`main.ipynb` contains the full pipeline from loading the data to training, validation, inference and visualization.

## About

**Type:** Solo project  
**Date:** September 2025  
**Task:** Multiclass semantic image segmentation