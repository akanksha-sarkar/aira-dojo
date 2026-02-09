import os
from pathlib import Path
import shutil
import yaml
import cv2
import numpy as np
import re
from pathlib import Path
import cv2
import numpy as np
from typing import Iterable, Tuple, Dict, Any, List
from torch.utils.data import Dataset
import tifffile as tiff
from src.util import _read_tif_any, _as_rgb, _extract_index, link_or_copy

class CellSegDataset(Dataset):
    """
    General-purpose segmentation dataset that pairs images and masks by numeric index.

    Returns:
        image: H x W x 3 (RGB)
        mask:  H x W (uint16 instance IDs) OR None if require_masks=False and missing
        meta:  dict with filename info
    """

    def __init__(
        self,
        data_dir,
        ann_dir,
        split: str = "train",
        require_masks: bool = True,
        image_glob: str = "*.tif",
        mask_glob: str = "*.tif",
    ):
        self.data_dir = Path(data_dir)
        self.ann_dir = Path(ann_dir)
        self.split = split
        self.require_masks = require_masks

        self.img_dir = self.data_dir
        self.mask_dir = self.ann_dir

        img_paths = sorted(self.img_dir.glob(image_glob))
        if not img_paths:
            raise RuntimeError(f"No images found in {self.img_dir} (glob={image_glob})")

        mask_paths = sorted(self.mask_dir.glob(mask_glob))
        if not mask_paths and require_masks:
            raise RuntimeError(f"No masks found in {self.mask_dir} (glob={mask_glob})")

        # Build index->path maps
        # Path.glob() returns Path objects - use them directly as-is
        # Don't resolve() - it can break symlinks or follow them to non-existent targets
        # The paths from glob() work fine when accessed directly
        img_by_idx: Dict[int, Path] = {}
        for p in img_paths:
            idx = _extract_index(p.stem)
            if idx is None:
                continue
            # Use path directly from glob() - it's already correct
            # If it's relative, make it absolute by combining with img_dir (but don't resolve)
            if p.is_absolute():
                img_by_idx[idx] = p
            else:
                # Combine but don't resolve - keep symlinks intact
                img_by_idx[idx] = self.img_dir / p

        mask_by_idx: Dict[int, Path] = {}
        for p in mask_paths:
            idx = _extract_index(p.stem)
            if idx is None:
                continue
            # Use path directly from glob() - it's already correct
            # If it's relative, make it absolute by combining with mask_dir (but don't resolve)
            if p.is_absolute():
                mask_by_idx[idx] = p
            else:
                # Combine but don't resolve - keep symlinks intact
                mask_by_idx[idx] = self.mask_dir / p

        if not img_by_idx:
            raise RuntimeError(
                f"Found images in {self.img_dir} but none matched an index pattern (e.g., ...0000.tif)."
            )

        self.pairs: List[Tuple[int, Path, Optional[Path]]] = []
        for idx in sorted(img_by_idx.keys()):
            img_p = img_by_idx[idx]
            m_p = mask_by_idx.get(idx, None)

            if self.require_masks and m_p is None:
                # Hard fail early so you don't discover this deep in training.
                raise RuntimeError(
                    f"Missing mask for image idx={idx}: image={img_p.name}. "
                    f"Expected a mask with the same trailing index in {self.mask_dir} "
                    f"(e.g., man_seg{idx:04d}.tif or mask{idx:04d}.tif)."
                )

            self.pairs.append((idx, img_p, m_p))

        if not self.pairs:
            raise RuntimeError("No (image, mask) pairs could be formed.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        idx, img_path, mask_path = self.pairs[i]

        # --- Load image ---
        img = _read_tif_any(img_path)
        img = _as_rgb(img)

        # --- Load mask ---
        mask = None
        if mask_path is not None:
            m = _read_tif_any(mask_path)
            if m.ndim == 3: # Some GT masks can be saved as HxWx1
                m = m[..., 0]
            mask = m.astype(np.uint16)

        meta = {
            "index": idx,
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path is not None else None,
        }

        return img, mask, meta

    def generate_yolo_dataset(self, yolo_dir: str | Path, *, class_id: int = 0, class_name: str = "cell", write_empty_labels: bool = True) -> str:
        """
        Generate Ultralytics YOLOv8-seg dataset for this dataset's split.

        Writes:
            yolo_dir/
            images/<split>/*.png
            labels/<split>/*.txt
            data.yaml

        Returns:
            str path to yolo_dir/data.yaml
        """
        yolo_dir = Path(yolo_dir)
        split = self.split

        print(f"Building YOLO-seg dataset at {yolo_dir} (split={split})...")

        yolo_image_dir = yolo_dir / "images" / split
        yolo_label_dir = yolo_dir / "labels" / split
        yolo_image_dir.mkdir(parents=True, exist_ok=True)
        yolo_label_dir.mkdir(parents=True, exist_ok=True)

        num_written = 0

        for idx, img_path, mask_path in self.pairs:
            # Try to read the file - if it fails, try reading directly from img_dir
            # This handles cases where symlinks might be broken but the file exists
            try:
                # ---------- IMAGE EXPORT (force 3-channel PNG) ----------
                img = _read_tif_any(img_path)
            except Exception as e:
                # If it's a broken symlink, try reading the symlink target directly
                if img_path.is_symlink():
                    try:
                        # Get the symlink target
                        target = img_path.readlink()
                        # Make target absolute if it's relative
                        if not target.is_absolute():
                            target = img_path.parent / target
                        # Try to read the target (resolve it to get the actual file)
                        resolved_target = target.resolve()
                        img = _read_tif_any(resolved_target)
                    except Exception as e2:
                        # If reading target also fails, provide detailed error
                        target_str = str(img_path.readlink()) if img_path.is_symlink() else 'N/A'
                        raise RuntimeError(
                            f"Failed to read symlink image at {img_path}\n"
                            f"  Symlink target: {target_str}\n"
                            f"  Original error: {e}\n"
                            f"  Target read error: {e2}\n"
                            f"  Image directory: {self.img_dir}\n"
                        ) from e2
                else:
                    # Not a symlink, just raise the original error
                    raise RuntimeError(
                        f"Failed to read image at {img_path}\n"
                        f"  Error: {e}\n"
                        f"  Image directory: {self.img_dir}\n"
                        f"  Path exists: {img_path.exists()}\n"
                    ) from e
            img = _as_rgb(img)

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            out_img_name = f"{img_path.stem}.png"
            out_img_path = yolo_image_dir / out_img_name

            if not out_img_path.exists():
                cv2.imwrite(str(out_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # ---------- LABEL GENERATION ----------
            label_path = yolo_label_dir / f"{img_path.stem}.txt"

            if mask_path is not None and Path(mask_path).exists():
                iou, dice = mask_to_yolo_and_iou(
                    mask_path=mask_path,
                    txt_path=label_path,
                    class_id=class_id,
                )
            else:
                if write_empty_labels:
                    label_path.touch()
                else:
                    continue

            num_written += 1

        # ---------- WRITE data.yaml ----------
        data_yaml_path = yolo_dir / "data.yaml"
        yaml_config = {
            "path": str(yolo_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {class_id: class_name},
        }

        if data_yaml_path.exists():
            try:
                with open(data_yaml_path, "r") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                existing = {}
            existing.update(yaml_config)
            yaml_config = existing

        with open(data_yaml_path, "w") as f:
            yaml.safe_dump(yaml_config, f, sort_keys=False)

        self.yolo_dir = yolo_dir
        self.yolo_yaml_path = str(data_yaml_path)

        print(f"Done. Wrote {num_written} samples to split={split}.")
        return str(data_yaml_path)


    def get_yolo_dataset(self) -> str:
        if not hasattr(self, "yolo_yaml_path"):
            raise RuntimeError("YOLO dataset has not been generated yet. Call generate_yolo_dataset(...) first.")
        return self.yolo_yaml_path

def collate_fn(batch):
    imgs, masks, metas = zip(*batch)
    return list(imgs), list(masks), list(metas)

def _ensure_rgb(img: np.ndarray, name: str = "") -> np.ndarray:
    """Ensure image is 3-channel uint8 RGB (for YOLO)."""
    if img is None:
        raise ValueError(f"Image {name} could not be read (cv2.imread returned None).")

    if img.ndim == 2:
        # grayscale → 3-channel
        img = cv2.merge([img, img, img])
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = cv2.merge([img[..., 0], img[..., 0], img[..., 0]])
        elif img.shape[2] == 3:
            pass  # already OK
        else:
            raise ValueError(f"Image {name} has unexpected shape {img.shape}")
    else:
        raise ValueError(f"Image {name} has unexpected ndim={img.ndim}")

    # Make sure type is uint8 (what YOLO expects)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def load_validation_dataset(val_root) -> Iterable[Tuple[Path, np.ndarray]]:
    """
    API-like helper to load a validation dataset in a YOLO-friendly way.

    Args:
        val_root: Path to validation root.
                  Can be either:
                    .../val/01
                  or
                    .../val/01/img

    Yields:
        (img_path, img_rgb) where img_rgb is a 3-channel uint8 numpy array.
    """
    val_root = Path(val_root)

    # If val_root itself has no tif images, assume there's an 'img' subfolder
    candidates = list(val_root.glob("*.tif"))
    if not candidates:
        img_dir = val_root / "img"
    else:
        img_dir = val_root

    if not img_dir.exists():
        raise FileNotFoundError(f"Could not find validation image directory at {img_dir}")

    for img_path in sorted(img_dir.glob("*.tif")):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img_rgb = _ensure_rgb(img, name=img_path.name)
        yield img_path, img_rgb

def mask_to_yolo_and_iou(mask_path: Path, txt_path: Path, class_id: int = 0):
    """
    Convert mask → YOLO polygon .txt
    Also compute IoU and Dice between mask and reconstructed polygons.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[WARN] Cannot read mask {mask_path}")
        return None, None
    if mask.ndim == 3:
        mask = mask[..., 0]

    h, w = mask.shape
    bin_mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    recon_mask = np.zeros_like(bin_mask, dtype=np.uint8)

    for c in contours:
        if len(c) < 3:
            continue
        eps = 0.001 * cv2.arcLength(c, True) # simplify polygon
        c_simpl = cv2.approxPolyDP(c, eps, True)
        if c_simpl.shape[0] < 3:
            continue
        coords = [] # convert contour to YOLO format 
        for (x, y) in c_simpl.squeeze():
            coords.append(x / w)
            coords.append(y / h)

        if len(coords) >= 6:
            lines.append(str(class_id) + " " + " ".join(f"{v:.6f}" for v in coords))
        cv2.fillPoly(recon_mask, [c_simpl.astype(np.int32)], 1)

    if not lines:
        return None, None
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    # Some sanity checks for polygon approximation quality
    intersection = np.logical_and(bin_mask == 1, recon_mask == 1).sum()
    union = np.logical_or(bin_mask == 1, recon_mask == 1).sum()
    if union == 0:
        iou = 1.0
        dice = 1.0
    else:
        iou = intersection / union
        dice = (2 * intersection) / (bin_mask.sum() + recon_mask.sum() + 1e-8)
    return float(iou), float(dice)


def create_yolo_dataset(base_dir, output_dir="yolo_dataset"):
    """
    Convert:
        base_dir/img/   with names like t1707.tif
        base_dir/seg/   with names like mask1707.tif

    → YOLO dataset:
        output_dir/images/train/
        output_dir/labels/train/
        output_dir/data.yaml

    Also prints aggregate IoU and Dice scores across all masks.
    """

    base_dir = Path(base_dir)
    img_src = base_dir / "img"
    mask_src = base_dir / "seg"

    out = Path(output_dir)
    img_dst = out / "images" / "train"
    lbl_dst = out / "labels" / "train"

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    ious, dices = [], []
    copied_images = set()

    # mask1707.tif  ->  t1707.tif
    for mask_file in sorted(mask_src.iterdir()):
        if mask_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            continue

        m = re.match(r"mask(\d+)", mask_file.stem)
        if not m:
            print(f"[WARN] Mask filename not in expected format: {mask_file.name}")
            continue

        num = m.group(1)
        img_name = f"t{num}{mask_file.suffix}"
        img_file = img_src / img_name

        if not img_file.exists():
            print(f"[WARN] No image for mask {mask_file.name} (looked for {img_name})")
            continue

        # Copy+convert image to RGB only once
        if img_file.name not in copied_images:
            img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"[WARN] Cannot read {img_file}")
                continue

            # --- Convert grayscale → RGB ---
            if img.ndim == 2:
                img = cv2.merge([img, img, img])
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.merge([img[..., 0], img[..., 0], img[..., 0]])
            elif img.ndim == 3 and img.shape[2] == 3:
                pass  # already OK
            else:
                print(f"[WARN] Unexpected image shape {img.shape} for {img_file.name}")
                continue

            cv2.imwrite(str(img_dst / img_file.name), img)
            copied_images.add(img_file.name)

        # Build YOLO polygon label
        txt_path = lbl_dst / (img_file.stem + ".txt")
        iou, dice = mask_to_yolo_and_iou(mask_file, txt_path, class_id=0)

        if iou is not None:
            ious.append(iou)
            dices.append(dice)
        else:
            print(f"{mask_file.name}: mask has no objects, skipping metrics")

    # Summary
    if ious:
        print("\n=== Polygon Approximation Quality ===")
        print(f"Masks processed: {len(ious)}")
        print(f"IoU  mean={np.mean(ious):.4f}, min={np.min(ious):.4f}, max={np.max(ious):.4f}")
        print(f"Dice mean={np.mean(dices):.4f}, min={np.min(dices):.4f}, max={np.max(dices):.4f}")
    else:
        print("No valid masks found.")

    # data.yaml
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": ["cell"]
    }

    with open(out / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    print(f"\nYOLO dataset created at: {out.resolve()}")

########################################################


########################################################