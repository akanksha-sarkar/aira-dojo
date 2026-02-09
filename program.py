import torchvision
import cv2
import torch
from torch.utils.data import DataLoader
from wildfin.src.dataset import collate_fn
from torchvision.ops import batched_nms
from ultralytics import YOLO

def finetune(train_dataset, val_dataset):
    data_yaml = train_dataset.get_yolo_dataset()
    model = YOLO("yolo11x.pt")
    model.train(data=data_yaml, epochs=50, imgsz=1920, batch=2, augment=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    coco_results = []
    for images, _, img_ids in val_loader:
        img = images[0].cpu().numpy().transpose(1, 2, 0)
        img_bgr = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
        all_boxes, all_scores, all_classes = [], [], []
        preds = model.predict(source=img_bgr, augment=True, imgsz=1920, conf=0.001, iou=0.5)
        for p in preds:
            all_boxes.append(p.boxes.xyxy.cpu())
            all_scores.append(p.boxes.conf.cpu())
            all_classes.append(p.boxes.cls.cpu().long())
        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            classes = torch.cat(all_classes, dim=0)
            keep = batched_nms(boxes, scores, classes, iou_threshold=0.6)
            for i in keep:
                x1, y1, x2, y2 = boxes[i].tolist()
                w, h = x2 - x1, y2 - y1
                coco_results.append({
                    "image_id": img_ids[0],
                    "category_id": train_dataset.pred_idx_to_category_id[int(classes[i])],
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(scores[i])
                })
    return coco_results