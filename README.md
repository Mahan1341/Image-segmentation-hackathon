# Semantic Image Segmentation

A solo machine-learning hackathon project for **3-class semantic image segmentation** of TIFF imagery.

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
| Classes | 3 (`0`, `1`, `2`; background + 2 foreground classes) |
| Input size | 512 × 512 |
| Optimizer | Adam |
| Initial learning rate | `1e-4` |
| Scheduler | ReduceLROnPlateau |
| Validation metric | multiclass IoU / Jaccard, background ignored |

The original notebook configured four output classes, but inspection of all 152 recovered historical mask files shows that the actual label set is only `{0, 1, 2}`. The cleaned pipeline therefore uses three classes and validates mask labels at load time.

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
├── inspect_dataset.py  # diagnostics for recovered GeoTIFF pairs
├── prepare_dataset.py  # rebuilds a verified paired subset locally
├── requirements.txt
├── .gitignore
└── README.md
```

`main.ipynb` is preserved as the original project artifact. `train.py` is the recommended entry point for rerunning the experiment now.

## Dataset recovery and layout

The dataset itself is intentionally not stored in the current working tree. A historical repository snapshot contains 158 TIFF images and 152 TIFF masks, but many filenames do not form trustworthy pairs.

To avoid fabricating correspondences, `prepare_dataset.py` keeps only image/mask pairs with a unique matching GeoTIFF signature (dimensions, transform, CRS and bounds). In the recovered snapshot this yields **53 verified pairs**. With the fixed seed used by the script, they are split into **42 training pairs** and **11 validation pairs**.

Run:

```bash
python prepare_dataset.py
```

The verified subset is written locally as:

```text
data/paired/
├── train/
│   ├── image/
│   └── mask/
└── val/
    ├── image/
    └── mask/
```

Image and mask filenames match inside the rebuilt subset. The `data/` directory is ignored by Git.

## Installation

```bash
git clone https://github.com/Mahan1341/Image-segmentation-hackathon.git
cd Image-segmentation-hackathon
pip install -r requirements.txt
```

## Training

After preparing the verified dataset, run:

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

After rerunning the experiment on the verified recovered subset, this section will include:

- best validation IoU;
- training/validation loss curves;
- validation IoU curve;
- qualitative image / ground-truth / prediction examples.

Because only 53 historical pairs could be verified unambiguously, rerun metrics should be interpreted as a reconstruction of the original experiment pipeline rather than as a claim about the full original hackathon dataset.

## Original post-processing

The hackathon notebook also contains a post-processing stage that applies per-class morphological opening/closing and removes small connected components before exporting final TIFF masks. This remains in `main.ipynb` as part of the original solution.

## Tech stack

Python · PyTorch · segmentation-models-pytorch · Albumentations · Rasterio · TorchMetrics · NumPy · Matplotlib · OpenCV · SciPy

## About

**Type:** Solo hackathon project  
**Date:** September 2025  
**Task:** Multiclass semantic image segmentation
