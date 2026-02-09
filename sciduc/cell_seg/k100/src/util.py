import os
import shutil
import logging
from pathlib import Path
import re
import numpy as np
import cv2
from typing import Optional
import tifffile as tiff


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlinks:
        try:
            os.symlink(src.resolve(), dst)
            return
        except Exception:
            if not copy_if_link_fails:
                raise
    # fallback
    shutil.copy2(src, dst)


def _extract_index(stem: str) -> Optional[int]:
    """
    Extract numeric frame index from filenames like:
      t0000, man_seg0000, mask_0123, etc.
    """
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    return int(m.group(1))


def _as_rgb(img: np.ndarray) -> np.ndarray:
    """Convert grayscale/BGRA to RGB. If already RGB/BGR, make it RGB."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise RuntimeError(f"Unexpected image shape: {img.shape}")


def _read_tif_any(path: Path) -> np.ndarray:
    """Reliable TIFF reader with OpenCV fallback."""
    try:
        import tifffile
        arr = tifffile.imread(str(path))
    except Exception:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if arr is None:
        raise RuntimeError(f"Failed to load file: {path}")
    return arr

def clear_dir(folder: str):
    """Delete all files/subfolders inside "folder" , but keep the folder itself."""
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)          
            elif os.path.isdir(path):
                shutil.rmtree(path)      
        except Exception as e:
            logging.error(f"Failed to delete {path}: {e}")

def rename_sequence(seq_dir):
    """
    Renames all .tif files in seq_dir to a continuous order:
        t0000.tif, t0001.tif, ...
    """

    seq_dir = Path(seq_dir)
    tif_files = sorted([p for p in seq_dir.glob("*.tif")])

    if not tif_files:
        logging.warning(f"[warn] No .tif files found in {seq_dir}")
        return

    logging.info(f"[rename] Found {len(tif_files)} files in {seq_dir}")

    # --- rename to temp names to avoid collisions
    temp_files = []
    for i, p in enumerate(tif_files):
        tmp = p.with_name(f"tmp_{i:04d}.tif")
        p.rename(tmp)
        temp_files.append(tmp)

    # --- rename to final sequential names
    for i, tmp in enumerate(sorted(temp_files, key=lambda p: p.name)):
        new_name = seq_dir / f"t{i:04d}.tif"
        tmp.rename(new_name)

    logging.info(f"[rename] Renamed {len(tif_files)} files to contiguous tXXXX.tif names.")
