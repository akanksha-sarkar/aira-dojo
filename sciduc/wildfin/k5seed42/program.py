import os
import torch
import yaml
import shutil
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from ultralytics import YOLO
from PIL import Image

# The ObjectDetectionDataset and collate_fn are provided by the environment.
from src.dataset import collate_fn


def finetune(train_dataset, val_dataset):
    """
    This function improves upon the state-of-the-art slice-based training pipeline by refining the inference and fusion stages.
    1. Slices the training images and annotations into smaller, high-resolution patches.
    2. Trains a large DEtection TRansformer (RT-DETR-L) model on this new sliced dataset with copy-paste augmentation.
    3. Performs overlapping tiled inference on the original high-resolution validation images, applying a stricter confidence threshold to pre-filter noise.
    4. Fuses the tiled predictions using Non-Maximum Suppression (NMS) with an optimized IoU threshold for better precision.
    """

    # --- 1. Slice-Based Training Data Preparation ---

    # Define slicing parameters
    slice_size = 1280
    overlap = 256
    stride = slice_size - overlap
    
    # Create directories for the sliced dataset
    sliced_data_dir = "./sliced_data"
    if os.path.exists(sliced_data_dir):
        shutil.rmtree(sliced_data_dir)
    print("Creating sliced data directories...")
    train_images_dir = os.path.join(sliced_data_dir, "images", "train")
    train_labels_dir = os.path.join(sliced_data_dir, "labels", "train")
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    print(len(train_dataset), "training images to slice.")
    print("Slicing training data...")
    for i in range(len(train_dataset)):
        image_tensor, target, img_id = train_dataset[i]

        image = torchvision.transforms.ToPILImage()(image_tensor)
        W, H = image.size

        boxes = target["boxes"].numpy()  # [x, y, w, h] COCO format
        labels = target["class_labels"].numpy()

        coco_cats = train_dataset.coco.cats
        coco_cat_ids = sorted(coco_cats.keys())
        cat_id_to_yolo_idx = {cat_id: i for i, cat_id in enumerate(coco_cat_ids)}

        img_info = train_dataset.coco.loadImgs(img_id)[0]
        base_filename = os.path.splitext(img_info["file_name"])[0]

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                x_end = min(x + slice_size, W)
                y_end = min(y + slice_size, H)
                tile_w = x_end - x
                tile_h = y_end - y

                if tile_w != slice_size or tile_h != slice_size:
                    continue

                tile_img = image.crop((x, y, x_end, y_end))
                tile_filename = f"{base_filename}_{y}_{x}"

                yolo_labels = []

                for box, label in zip(boxes, labels):
                    box_x, box_y, box_w, box_h = box
                    box_x2, box_y2 = box_x + box_w, box_y + box_h

                    inter_x1 = max(box_x, x)
                    inter_y1 = max(box_y, y)
                    inter_x2 = min(box_x2, x_end)
                    inter_y2 = min(box_y2, y_end)

                    inter_w = inter_x2 - inter_x1
                    inter_h = inter_y2 - inter_y1

                    if inter_w > 10 and inter_h > 10:
                        new_x = inter_x1 - x
                        new_y = inter_y1 - y
                        new_w = inter_w
                        new_h = inter_h

                        center_x = (new_x + new_w / 2) / slice_size
                        center_y = (new_y + new_h / 2) / slice_size
                        norm_w = new_w / slice_size
                        norm_h = new_h / slice_size

                        yolo_idx = cat_id_to_yolo_idx[int(label)]
                        yolo_labels.append(
                            f"{yolo_idx} {center_x} {center_y} {norm_w} {norm_h}"
                        )

                if yolo_labels:
                    tile_img.save(
                        os.path.join(train_images_dir, f"{tile_filename}.jpg")
                    )
                    with open(
                        os.path.join(train_labels_dir, f"{tile_filename}.txt"), "w"
                    ) as f:
                        f.write("\n".join(yolo_labels))

    # --- 2. Create YAML and Train with Copy-Paste Augmentation ---

    yolo_yaml_path = os.path.join(sliced_data_dir, "dataset.yaml")
    class_names = [coco_cats[cat_id]["name"] for cat_id in coco_cat_ids]

    yaml_content = {
        "train": os.path.abspath(train_images_dir),
        "val": os.path.abspath(
            train_images_dir
        ),  # Use train set for validation during training
        "nc": len(class_names),
        "names": class_names,
    }

    with open(yolo_yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    print("Training RT-DETR-L on sliced data with copy-paste augmentation...")
    model = YOLO("rtdetr-l.pt")
    model.train(
        data=yolo_yaml_path,
        epochs=10,
        imgsz=1280,
        batch=8,
        project="./yolo_runs",
        name="train_rtdetr_sliced_copypaste_refined",
        patience=10,
        verbose=False,
        workers=8,
        mosaic=0.0,
        copy_paste=0.5,
    )

    # --- 3. Tiled Inference on Validation Set with Refined Thresholds ---

    print("Performing tiled inference with refined thresholds...")
    inference_model_path = os.path.join(
        "./yolo_runs/train_rtdetr_sliced_copypaste_refined/weights/best.pt"
    )
    inference_model = YOLO(inference_model_path)

    coco_results = []

    # Use validation dataset to create the mapping to ensure correctness
    val_coco_cats = val_dataset.coco.cats
    val_coco_cat_ids = sorted(val_coco_cats.keys())
    yolo_idx_to_cat_id = {i: cat_id for i, cat_id in enumerate(val_coco_cat_ids)}

    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    for images, targets, img_ids in val_dataloader:
        img_id = img_ids[0]
        image_tensor = images[0]
        image = torchvision.transforms.ToPILImage()(image_tensor)
        W, H = image.size

        tiles = []
        tile_coords = []
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                x_end = min(x + slice_size, W)
                y_end = min(y + slice_size, H)
                tile_w = x_end - x
                tile_h = y_end - y

                if tile_w < slice_size or tile_h < slice_size:
                    padded_tile = Image.new(
                        "RGB", (slice_size, slice_size), (114, 114, 114)
                    )
                    tile_img = image.crop((x, y, x_end, y_end))
                    padded_tile.paste(tile_img, (0, 0))
                    tiles.append(padded_tile)
                else:
                    tile_img = image.crop((x, y, x_end, y_end))
                    tiles.append(tile_img)

                tile_coords.append((x, y))

        if not tiles:
            continue

        all_boxes = []
        all_scores = []
        all_labels = []

        # Improvement 1: Stricter confidence threshold during prediction
        results = inference_model.predict(
            tiles, verbose=False, imgsz=1280, batch=16, conf=0.1
        )

        for i, result in enumerate(results):
            x_offset, y_offset = tile_coords[i]
            boxes = result.boxes
            for j in range(len(boxes)):
                box = boxes[j]
                xyxy = box.xyxy.cpu().numpy()[0]
                conf = box.conf.cpu().numpy()[0]
                cls_idx = int(box.cls.cpu().numpy()[0])

                x1, y1, x2, y2 = xyxy

                orig_w = min(slice_size, W - x_offset)
                orig_h = min(slice_size, H - y_offset)
                if x1 >= orig_w or y1 >= orig_h:
                    continue

                global_x1 = x1 + x_offset
                global_y1 = y1 + y_offset
                global_x2 = min(x2 + x_offset, W)
                global_y2 = min(y2 + y_offset, H)

                all_boxes.append([global_x1, global_y1, global_x2, global_y2])
                all_scores.append(conf)
                all_labels.append(cls_idx)

        if not all_boxes:
            continue

        # --- 4. Fusion with Non-Maximum Suppression (NMS) with Optimized IoU ---
        unique_labels = np.unique(all_labels)
        for label in unique_labels:
            class_indices = [k for k, l in enumerate(all_labels) if l == label]
            class_boxes = torch.tensor(
                [all_boxes[k] for k in class_indices], dtype=torch.float32
            )
            class_scores = torch.tensor(
                [all_scores[k] for k in class_indices], dtype=torch.float32
            )

            if len(class_boxes) == 0:
                continue

            # Improvement 2: Optimized NMS IoU threshold
            nms_indices = torchvision.ops.nms(
                class_boxes, class_scores, iou_threshold=0.5
            )

            for idx in nms_indices:
                final_box = class_boxes[idx].numpy()
                final_score = class_scores[idx].item()
                x1, y1, x2, y2 = final_box
                width = x2 - x1
                height = y2 - y1

                coco_cat_id = yolo_idx_to_cat_id[int(label)]

                coco_results.append(
                    {
                        "image_id": img_id,
                        "category_id": coco_cat_id,
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(final_score),
                    }
                )

    return coco_results