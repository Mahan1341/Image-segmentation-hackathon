from collections import defaultdict
from pathlib import Path

import rasterio


IMAGE_DIRS = [Path("train/train/image"), Path("train/val/image")]
MASK_DIRS = [Path("train/train/mask"), Path("train/val/mask")]


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
        return (
            src.width,
            src.height,
            transform,
            crs,
            bounds,
        )


def main():
    images = collect(IMAGE_DIRS)
    masks = collect(MASK_DIRS)

    image_by_sig = defaultdict(list)
    mask_by_sig = defaultdict(list)

    for path in images:
        image_by_sig[signature(path)].append(path)
    for path in masks:
        mask_by_sig[signature(path)].append(path)

    unique_pairs = []
    ambiguous = []
    image_only = []

    for sig, image_paths in image_by_sig.items():
        mask_paths = mask_by_sig.get(sig, [])
        if len(image_paths) == 1 and len(mask_paths) == 1:
            unique_pairs.append((image_paths[0], mask_paths[0]))
        elif mask_paths:
            ambiguous.append((image_paths, mask_paths))
        else:
            image_only.extend(image_paths)

    matched_mask_paths = {
        mask
        for _, mask in unique_pairs
    }
    for _, masks_for_sig in ambiguous:
        matched_mask_paths.update(masks_for_sig)

    mask_only = [path for path in masks if path not in matched_mask_paths]

    print(f"Images:                 {len(images)}")
    print(f"Masks:                  {len(masks)}")
    print(f"Unique image signatures:{len(image_by_sig)}")
    print(f"Unique mask signatures: {len(mask_by_sig)}")
    print(f"Unambiguous pairs:      {len(unique_pairs)}")
    print(f"Ambiguous signatures:   {len(ambiguous)}")
    print(f"Images without match:   {len(image_only)}")
    print(f"Masks without match:    {len(mask_only)}")

    print("\nFirst unambiguous pairs:")
    for image, mask in unique_pairs[:20]:
        print(f"  {image}  <->  {mask}")

    if ambiguous:
        print("\nFirst ambiguous signatures:")
        for image_paths, mask_paths in ambiguous[:5]:
            print("  images:", ", ".join(str(path) for path in image_paths[:5]))
            print("  masks: ", ", ".join(str(path) for path in mask_paths[:5]))

    if image_only:
        print("\nFirst images without metadata match:")
        for path in image_only[:20]:
            print(f"  {path}")

    if mask_only:
        print("\nFirst masks without metadata match:")
        for path in mask_only[:20]:
            print(f"  {path}")


if __name__ == "__main__":
    main()
