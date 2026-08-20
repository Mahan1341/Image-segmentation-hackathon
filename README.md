# Semantic Image Segmentation

A solo machine-learning hackathon project for **4-class semantic image segmentation** of TIFF imagery.

The original solution was developed in **September 2025** using **DeepLabV3+** with an **EfficientNet-B5** encoder. The repository now contains both the original hackathon notebook and a cleaned training script for reproducible reruns.

## What the project covers

- TIFF image/mask loading with Rasterio
- training and validation datasets
- Albumentations-based augmentation
- DeepLabV3+ model configuration
- ImageNet-pretrained EfficientNet-B5 encoder
- combined Dice + Focal loss
- IoU/Jaccard validation
- checkpoint selection by validation IoU
- prediction visualization
- training-curve export
- morphological mask post-processing in the original notebook

## Model

| Component | Configuration |
| --- | --- |
| Architecture | DeepLabV3+ |
| Encoder | EfficientNet-B5 (`timm-efficientnet-b5`) |
| Encoder weights | ImageNet pretrained |
| Classes | 4 (background + 3 foreground classes) |
| Input size | 512 × 512 |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Scheduler | ReduceLROnPlateau |
| Validation metric | multiclass IoU / Jaccard, background ignored |

The cleaned training pipeline keeps the model output as **raw logits**. Dice and Focal losses then handle the logits internally instead of applying softmax inside the segmentation model.

## Data augmentation

The training pipeline uses Albumentations with:

- horizontal and vertical flips
- random 90° rotations
- affine translation, scale and rotation
- brightness/contrast changes
- Gaussian noise and blur
- grid distortion
- coarse dropout
- ImageNet normalization

Validation uses deterministic resizing and normalization only.

## Repository structure

```text
.
├── main.ipynb          # original hackathon notebook
├── train.py            # cleaned reproducible training pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

`main.ipynb` is preserved as the original project artifact. `train.py` is the recommended entry point for rerunning the experiment now.

## Dataset layout

The dataset itself is intentionally not stored in the current working tree. The training script expects:

```text
train/
├── train/
│   ├── image/
│   │   └── *.tif
│   └── mask/
│       └── *.tif
└── val/
    ├── image/
    │   └── *.tif
    └── mask/
        └── *.tif
```

Image and mask filenames must match.

## Installation

```bash
git clone https://github.com/Mahan1341/Image-segmentation-hackathon.git
cd Image-segmentation-hackathon
pip install -r requirements.txt
```

## Training

Run the cleaned pipeline with:

```bash
python train.py
```

Useful options:

```bash
python train.py --epochs 50 --batch-size 4 --lr 1e-4
```

On Windows, the script defaults to `num_workers=0` to avoid multiprocessing-related DataLoader failures. This can be overridden manually:

```bash
python train.py --num-workers 2
```

The best model is saved to:

```text
best_model.pth
```

Generated visualizations are saved to:

```text
artifacts/training_curves.png
artifacts/prediction_examples.png
```

Model checkpoints are excluded from Git because of their size. Small result images can be committed after a reproducible run and used directly in this README.

## Results

The current repository does **not** claim a validation score because the previously committed notebook outputs did not contain a completed training run. The cleaned pipeline is intended to make the result reproducible rather than inventing a metric retrospectively.

After rerunning the experiment, this section will include:

- best validation IoU;
- training/validation loss curves;
- validation IoU curve;
- qualitative image / ground-truth / prediction examples.

## Original post-processing

The hackathon notebook also contains a post-processing stage that applies per-class morphological opening/closing and removes small connected components before exporting final TIFF masks. This remains in `main.ipynb` as part of the original solution.

## Tech stack

Python · PyTorch · segmentation-models-pytorch · Albumentations · Rasterio · TorchMetrics · NumPy · Matplotlib · OpenCV · SciPy

## About

**Type:** Solo hackathon project  
**Date:** September 2025  
**Task:** Multiclass semantic image segmentation
