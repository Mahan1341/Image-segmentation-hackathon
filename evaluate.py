import argparse

import torch
from torch.utils.data import DataLoader

from train import (
    NUM_CLASSES,
    SegmentationDataset,
    build_transforms,
    compute_band_statistics,
    create_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved segmentation checkpoint")
    parser.add_argument("--train-images", default="data/paired/train/image")
    parser.add_argument("--val-images", default="data/paired/val/image")
    parser.add_argument("--val-masks", default="data/paired/val/mask")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = compute_band_statistics(args.train_images)
    _, val_transform = build_transforms(args.image_size, mean, std)
    dataset = SegmentationDataset(args.val_images, args.val_masks, val_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = create_model(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        predictions = model(images).argmax(dim=1)

        valid = (masks >= 0) & (masks < NUM_CLASSES)
        encoded = NUM_CLASSES * masks[valid] + predictions[valid]
        confusion += torch.bincount(
            encoded,
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)

    confusion = confusion.cpu()
    print(f"Device: {device}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion.numpy())

    ious = []
    for class_id in range(NUM_CLASSES):
        tp = confusion[class_id, class_id].item()
        fp = confusion[:, class_id].sum().item() - tp
        fn = confusion[class_id, :].sum().item() - tp
        denominator = tp + fp + fn
        iou = tp / denominator if denominator else float("nan")
        ious.append(iou)
        label = "background" if class_id == 0 else f"class {class_id}"
        print(f"IoU {label}: {iou:.4f}")

    foreground = [iou for iou in ious[1:] if iou == iou]
    mean_foreground = sum(foreground) / len(foreground)
    print(f"Mean foreground IoU: {mean_foreground:.4f}")


if __name__ == "__main__":
    main()
