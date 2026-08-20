import argparse
import os
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm


NUM_CLASSES = 4
IMAGE_SIZE = 512
ENCODER = "timm-efficientnet-b5"
ENCODER_WEIGHTS = "imagenet"
DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 2


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transform=None) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.images = sorted(self.images_dir.glob("*.tif"))

        if not self.images:
            raise FileNotFoundError(f"No .tif images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        mask_path = self.masks_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}: {mask_path}")

        with rasterio.open(image_path) as src:
            image = src.read().transpose(1, 2, 0).astype(np.float32)

        if image.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel image, got shape {image.shape} for {image_path.name}"
            )

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.long()


def build_transforms(image_size: int):
    train_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.2, 0.2),
                rotate=(-45, 45),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=0.5,
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.5,
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def create_model(device: torch.device) -> nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model.to(device)


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.7, gamma: float = 2.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.focal = smp.losses.FocalLoss(mode="multiclass", gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.dice_weight * dice_loss + (1.0 - self.dice_weight) * focal_loss


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    progress = tqdm(loader, desc="Train", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    metric = MulticlassJaccardIndex(
        num_classes=NUM_CLASSES,
        ignore_index=0,
        average="macro",
    ).to(device)

    progress = tqdm(loader, desc="Validation", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)
        predictions = logits.argmax(dim=1)

        metric.update(predictions, masks)
        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader), float(metric.compute().cpu())


def save_training_curves(train_losses, val_losses, val_ious, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="Train loss")
    axes[0].plot(val_losses, label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].legend()

    axes[1].plot(val_ious, label="Validation IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation IoU")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_prediction_examples(model, dataset, device, output_path: Path, num_samples: int = 4):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    fig, axes = plt.subplots(count, 3, figsize=(12, 4 * count), squeeze=False)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for row, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        display_image = image.permute(1, 2, 0).cpu().numpy()
        display_image = np.clip(display_image * std + mean, 0.0, 1.0)

        axes[row, 0].imshow(display_image)
        axes[row, 0].set_title("Image")
        axes[row, 1].imshow(true_mask.cpu().numpy(), vmin=0, vmax=NUM_CLASSES - 1)
        axes[row, 1].set_title("Ground truth")
        axes[row, 2].imshow(pred_mask, vmin=0, vmax=NUM_CLASSES - 1)
        axes[row, 2].set_title("Prediction")

        for column in range(3):
            axes[row, column].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the semantic segmentation model")
    parser.add_argument("--train-images", default="train/train/image")
    parser.add_argument("--train-masks", default="train/train/mask")
    parser.add_argument("--val-images", default="train/val/image")
    parser.add_argument("--val-masks", default="train/val/mask")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--artifacts-dir", default="artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print(f"Device: {device}")

    train_transform, val_transform = build_transforms(args.image_size)
    train_dataset = SegmentationDataset(args.train_images, args.train_masks, train_transform)
    val_dataset = SegmentationDataset(args.val_images, args.val_masks, val_transform)

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = create_model(device)
    loss_fn = CombinedLoss(dice_weight=0.7)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = -1.0
    checkpoint_path = Path(args.checkpoint)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_iou = validate(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"best_iou={best_iou:.4f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    artifacts_dir = Path(args.artifacts_dir)
    save_training_curves(
        train_losses,
        val_losses,
        val_ious,
        artifacts_dir / "training_curves.png",
    )
    save_prediction_examples(
        model,
        val_dataset,
        device,
        artifacts_dir / "prediction_examples.png",
    )

    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved visualizations: {artifacts_dir}")


if __name__ == "__main__":
    main()
