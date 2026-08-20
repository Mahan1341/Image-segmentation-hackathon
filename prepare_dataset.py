import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio


SOURCE_IMAGE_DIRS = [Path("train/train/image"), Path("train/val/image")]
SOURCE_MASK_DIRS = [Path("train/train/mask"), Path("train/val/mask")]
OUTPUT_ROOT = Path("data/paired")
SEED = 42
VAL_FRACTION = 0.2


def collect(paths):
    files = []
    for path in paths:
        files.extend(sorted(path.glob("*.tif")))
    return files


def signature(path: Path):
    with rasterio.open(path) as src:
        transform = tuple(round(float(value), 12) for value in src.transform)
        crs = src.crs.to_string() if src.crs is not None else None
        bounds = tuple(round(float(value), 12) for value in src.bounds)
        return src.width, src.height, transform, crs, bounds


def find_verified_pairs():
    images_by_signature = defaultdict(list)
    masks_by_signature = defaultdict(list)

    for path in collect(SOURCE_IMAGE_DIRS):
        images_by_signature[signature(path)].append(path)

    for path in collect(SOURCE_MASK_DIRS):
        masks_by_signature[signature(path)].append(path)

    pairs = []
    for sig, image_paths in images_by_signature.items():
        mask_paths = masks_by_signature.get(sig, [])
        if len(image_paths) == 1 and len(mask_paths) == 1:
            pairs.append((image_paths[0], mask_paths[0]))

    return sorted(pairs, key=lambda pair: str(pair[0]))


def mask_classes(path: Path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    return set(int(value) for value in np.unique(mask))


def summarize_split(name, pairs):
    class_presence = Counter()
    for _, mask_path in pairs:
        for class_id in mask_classes(mask_path):
            class_presence[class_id] += 1

    print(f"{name}: {len(pairs)} pairs")
    print(
        "  mask class presence: "
        + ", ".join(
            f"class {class_id} in {count}/{len(pairs)} masks"
            for class_id, count in sorted(class_presence.items())
        )
    )


def copy_split(name, pairs):
    image_dir = OUTPUT_ROOT / name / "image"
    mask_dir = OUTPUT_ROOT / name / "mask"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index, (image_path, mask_path) in enumerate(pairs):
        filename = f"{index:03d}.tif"
        shutil.copy2(image_path, image_dir / filename)
        shutil.copy2(mask_path, mask_dir / filename)


def main():
    pairs = find_verified_pairs()
    if not pairs:
        raise RuntimeError("No unambiguous image-mask pairs were found")

    print(f"Verified pairs found: {len(pairs)}")

    rng = random.Random(SEED)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)

    val_size = max(1, round(len(shuffled) * VAL_FRACTION))
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    copy_split("train", train_pairs)
    copy_split("val", val_pairs)

    summarize_split("Train", train_pairs)
    summarize_split("Validation", val_pairs)

    print(f"Prepared dataset written to: {OUTPUT_ROOT}")
    print("Only pairs with a unique matching GeoTIFF signature were included.")


if __name__ == "__main__":
    main()
