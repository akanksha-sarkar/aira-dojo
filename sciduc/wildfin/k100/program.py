import torchvision
import cv2
import torch
from torch.utils.data import DataLoader
from wildfin.src.dataset import collate_fn
from torchvision.ops import batched_nms
from ultralytics import YOLO

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
        data=train_dataset.get_yolo_dataset(), epochs=10, imgsz=1920
    )  # Increase epochs and imgsz, reduce batch if necessary
    # Create a mapping from the model's 0-indexed class IDs
    # back to the original COCO category IDs for correct evaluation.
    train_coco = train_dataset.coco
    cat_ids = sorted(train_coco.getCatIds())
    yolo_to_coco_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    coco_results = []
    for img_id in val_dataset.img_ids:
        img_info = val_dataset.coco.loadImgs(img_id)[0]
        img_path = os.path.join(val_dataset.data_dir, img_info["file_name"])

        # Perform inference at the same high resolution used for training.
        results = model.predict(img_path, imgsz=1920, verbose=False, device=0)

        result = results[0]  # Get results for the single image
        boxes = result.boxes

        # Process each detection
        for box in boxes:
            # Extract bounding box in xyxy format
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = coords

            # Convert to COCO's required xywh format
            bbox = [x1, y1, x2 - x1, y2 - y1]

            # Get class ID and confidence score
            yolo_class_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            # Map the YOLO class ID back to the original COCO category ID
            if yolo_class_id in yolo_to_coco_cat_id:
                coco_cat_id = yolo_to_coco_cat_id[yolo_class_id]

                # Append the formatted result
                coco_results.append(
                    {
                        "image_id": img_id,
                        "category_id": coco_cat_id,
                        "bbox": [round(c, 2) for c in bbox],
                        "score": round(conf, 4),
                    }
                )

    print(f"Generated {len(coco_results)} detections.")
    return coco_results