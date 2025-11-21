from src.dataset import collate_fn, ObjectDetectionDataset
from torch.utils.data import DataLoader
import cv2
import ultralytics
import torchvision
import torch
import os
from pathlib import Path
import json
from pycocotools.coco import COCO
import torchvision.transforms.functional as F


def finetune(train_dataset, val_dataset):
    """
    This function should produce a list of COCO format detection results for each image in the validation dataset.
    You can use whatever method you want to produce the detection results. Here is some useful information:
    - You can use the train_dataloader to get the images and targets for the training dataset.
    - You can use the val_dataloader to get the images and targets for the validation dataset.
    Args:
        train_dataset: ObjectDetectionDataset
        val_dataset: ObjectDetectionDataset
    Returns:
        coco_results: A list of COCO format detection results for each image in the validation dataset.
    """
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    # 1. Prepare YOLOv8 model and trainer
    model = ultralytics.YOLO(
        "yolov8s.pt"
    )  # Use a larger model variant, e.g., yolov8m.pt or yolov8l.pt if resources permit

    # 2. Train the model
    model.train(
        data=train_dataset.get_yolo_dataset(), epochs=5, imgsz=1920
    )  # Increase epochs and imgsz, reduce batch if necessary
    coco_results = []
    for _, _, img_ids in val_dataloader:
        img_paths = []
        for img_id in img_ids:
            info = val_dataset.coco.loadImgs(img_id)[0]
            img_paths.append(os.path.join(val_dataset.data_dir, info['file_name']))
        results = model.predict(source=img_paths, augment=True, imgsz=1920)
        for res, img_id in zip(results, img_ids):
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box.tolist()
                coco_results.append({
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score)
                })
    return coco_results