import argparse
import os
import random
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
from tqdm import tqdm


NUM_CLASSES = 3
INPUT_CHANNELS = 6
IMAGE_SIZE = 256
SEED = 42
BAND_NAMES = ("B02", "B03", "B04", "B08", "B11", "B12")
ENCODER = "timm-efficientnet-b5"
ENCODER_WEIGHTS = "imagenet"
DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 2


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

        if image.shape[2] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} channels, got shape {image.shape} "
                f"for {image_path.name}"
            )

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        if mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(
                f"Mask {mask_path.name} contains labels outside [0, {NUM_CLASSES - 1}]"
            )

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask.long()


def compute_band_statistics(images_dir: str):
    image_paths = sorted(Path(images_dir).glob("*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No .tif images found in {images_dir}")

    band_sum = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    band_sq_sum = np.zeros(INPUT_CHANNELS, dtype=np.float64)
    pixel_count = 0

    for path in image_paths:
        with rasterio.open(path) as src:
            image = src.read().astype(np.float64)

        if image.shape[0] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} channels, got {image.shape[0]} for {path.name}"
            )

        flattened = image.reshape(INPUT_CHANNELS, -1)
        band_sum += flattened.sum(axis=1)
        band_sq_sum += np.square(flattened).sum(axis=1)
        pixel_count += flattened.shape[1]

    mean = band_sum / pixel_count
    variance = band_sq_sum / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return tuple(mean.tolist()), tuple(std.tolist())


def compute_class_weights(masks_dir: str):
    mask_paths = sorted(Path(masks_dir).glob("*.tif"))
    if not mask_paths:
        raise FileNotFoundError(f"No .tif masks found in {masks_dir}")

    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for path in mask_paths:
        with rasterio.open(path) as src:
            mask = src.read(1).astype(np.int64)
        counts += np.bincount(mask.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]

    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(np.maximum(frequencies, 1e-12))
    weights /= weights.mean()
    return counts, frequencies, weights.astype(np.float32)


def build_transforms(image_size: int, mean, std):
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
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def create_model(device: torch.device) -> nn.Module:
    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=INPUT_CHANNELS,
        classes=NUM_CLASSES,
        activation=None,
    )
    return model.to(device)


class CombinedLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(logits, targets)
        ce_loss = self.cross_entropy(logits, targets)
        return self.dice_weight * dice_loss + (1.0 - self.dice_weight) * ce_loss


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


def update_confusion(confusion: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor):
    valid = (targets >= 0) & (targets < NUM_CLASSES)
    encoded = NUM_CLASSES * targets[valid] + predictions[valid]
    confusion += torch.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    ).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(confusion: torch.Tensor):
    ious = []
    for class_id in range(NUM_CLASSES):
        tp = confusion[class_id, class_id].item()
        fp = confusion[:, class_id].sum().item() - tp
        fn = confusion[class_id, :].sum().item() - tp
        denominator = tp + fp + fn
        ious.append(tp / denominator if denominator else float("nan"))

    foreground = [value for value in ious[1:] if not np.isnan(value)]
    mean_foreground_iou = float(np.mean(foreground)) if foreground else float("nan")
    return ious, mean_foreground_iou


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    progress = tqdm(loader, desc="Validation", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)
        predictions = logits.argmax(dim=1)

        update_confusion(confusion, masks, predictions)
        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    class_ious, mean_foreground_iou = iou_from_confusion(confusion)
    return running_loss / len(loader), class_ious, mean_foreground_iou


def save_training_curves(train_losses, val_losses, val_ious, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="Train loss")
    axes[0].plot(val_losses, label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss")
    axes[0].legend()

    axes[1].plot(val_ious, label="Mean foreground IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation foreground IoU")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def percentile_stretch(rgb: np.ndarray) -> np.ndarray:
    low = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    high = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    scale = np.maximum(high - low, 1e-6)
    return np.clip((rgb - low) / scale, 0.0, 1.0)


@torch.no_grad()
def save_prediction_examples(
    model,
    dataset,
    device,
    output_path: Path,
    mean,
    std,
    num_samples: int = 4,
):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = min(num_samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    fig, axes = plt.subplots(count, 3, figsize=(12, 4 * count), squeeze=False)

    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    for row, idx in enumerate(indices):
        image, true_mask = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        multispectral = image.permute(1, 2, 0).cpu().numpy()
        multispectral = multispectral * std_arr + mean_arr
        rgb = percentile_stretch(multispectral[..., [2, 1, 0]])

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title("B04/B03/B02")
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
    parser.add_argument("--train-images", default="data/paired/train/image")
    parser.add_argument("--train-masks", default="data/paired/train/mask")
    parser.add_argument("--val-images", default="data/paired/val/image")
    parser.add_argument("--val-masks", default="data/paired/val/mask")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Bands: {', '.join(BAND_NAMES)}")

    mean, std = compute_band_statistics(args.train_images)
    counts, frequencies, weights = compute_class_weights(args.train_masks)

    print("Training-band mean:", ", ".join(f"{value:.6f}" for value in mean))
    print("Training-band std: ", ", ".join(f"{value:.6f}" for value in std))
    print("Class pixels:      ", ", ".join(str(value) for value in counts))
    print("Class frequency:   ", ", ".join(f"{value:.4f}" for value in frequencies))
    print("Class weights:     ", ", ".join(f"{value:.4f}" for value in weights))

    train_transform, val_transform = build_transforms(args.image_size, mean, std)
    train_dataset = SegmentationDataset(args.train_images, args.train_masks, train_transform)
    val_dataset = SegmentationDataset(args.val_images, args.val_masks, val_transform)

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = create_model(device)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    loss_fn = CombinedLoss(class_weights=class_weights, dice_weight=0.5)
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
        val_loss, class_ious, mean_foreground_iou = validate(
            model, val_loader, loss_fn, device
        )
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(mean_foreground_iou)

        if mean_foreground_iou > best_iou:
            best_iou = mean_foreground_iou
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"iou_bg={class_ious[0]:.4f} | "
            f"iou_c1={class_ious[1]:.4f} | "
            f"iou_c2={class_ious[2]:.4f} | "
            f"fg_miou={mean_foreground_iou:.4f} | "
            f"best_fg_miou={best_iou:.4f}"
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
        mean,
        std,
    )

    print(f"Best mean foreground IoU: {best_iou:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved visualizations: {artifacts_dir}")


if __name__ == "__main__":
    main()
