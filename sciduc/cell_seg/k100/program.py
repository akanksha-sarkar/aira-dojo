import yaml
import shutil
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from src.dataset import create_yolo_dataset, CellSegDataset, collate_fn
import re
from typing import Dict, List, Optional, Tuple
import os


def cell_segmentation(train_dataset: "CellSegDataset", val_dataset: "CellSegDataset"):
    """
    Returns:
        seg_mask_list: list[np.ndarray] of uint16 instance-id masks for each val image
    """
    epochs = 2
    imgsz = 640
    batch = 8
    pretrained = "yolov8n-seg.pt"
    device = 0 if torch.cuda.is_available() else "cpu"
    
    cache_dir = Path("/cache")
    if cache_dir.exists() and os.access(cache_dir, os.W_OK):
        yolo_root = cache_dir / "yolo_cellseg"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)
    yolo_root.mkdir(parents=True, exist_ok=True)
    print(f"Using YOLO dataset directory: {yolo_root}")

    # -------------------------
    # 1) Export datasets directly to YOLO format
    # -------------------------

    train_yaml = train_dataset.generate_yolo_dataset(yolo_root)
    val_yaml = val_dataset.generate_yolo_dataset(yolo_root)
    data_yaml_path = Path(train_yaml)

    # -------------------------
    # 2) Remove stale Ultralytics cache files 
    # -------------------------
    for cache in (yolo_root / "labels").glob("*.cache"):
        cache.unlink(missing_ok=True)

    # -------------------------
    # 3) Train YOLO
    # -------------------------
    print("Training YOLO model...")
    model = YOLO(pretrained)
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        patience=5,
        project="yolo_runs",
        name="cell_seg_exp",
        exist_ok=True,
    )

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_model_path))

    # -------------------------
    # 4) Predict over val_dataset (DataLoader path unchanged)
    # -------------------------
    print("Predicting over val dataset...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    seg_mask_list = []
    for imgs, _, _ in val_loader:
        img = imgs[0]
        H, W = img.shape[:2]

        pred = best_model.predict(img, verbose=False)
        out_mask = np.zeros((H, W), dtype=np.uint16)

        if pred and pred[0].masks is not None and pred[0].masks.data is not None:
            masks = pred[0].masks.data  # (N, h, w)
            for j in range(masks.shape[0]):
                m = masks[j].detach().cpu().numpy() > 0.5
                if m.shape != (H, W):
                    m = cv2.resize(
                        m.astype(np.uint8),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                out_mask[m] = j + 1

        seg_mask_list.append(out_mask)

    return seg_mask_list


### Dummy main function for sanity testing
if __name__ == "__main__":
    train_dataset = CellSegDataset(
        data_dir="/share/j_sun/as2637/sciduc/cell_seg/k100/data/01/img",
        ann_dir="/share/j_sun/as2637/sciduc/cell_seg/k100/data/01/seg",
        split="train",
    )
    val_dataset = CellSegDataset(
        data_dir="/share/j_sun/as2637/sciduc/cell_seg/data/val/01/img",
        ann_dir="/share/j_sun/as2637/sciduc/cell_seg/data/val/01/GT/SEG",
        split="val",
    )

    seg_mask_list = cell_segmentation(train_dataset=train_dataset, val_dataset=val_dataset)
    print(len(seg_mask_list))