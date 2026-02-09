#!/usr/bin/env python3
"""
Ultralytics YOLO pose finetuning with aggressive augmentation and cosine LR schedule.
"""

import os
import time
from typing import Dict, Optional
from ultralytics import YOLO

def finetune(
    dataset_yaml: str,
    *,
    epochs: int = 50,
    imgsz: int = 640,
    model: str = "yolo11n-pose.pt",
    project_dir: Optional[str] = None,
    name: str = "weco-pose-aug-cos"
) -> Dict:
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
    start = time.time()
    yolo_model = YOLO(model)
    train_kwargs = {
        "data": dataset_yaml,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "device": 0 if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu",
        "project": project_dir or os.path.join(os.getcwd(), "runs"),
        "name": name,
        "exist_ok": True,
    }
    # Aggressive augmentation and cosine LR schedule
    train_kwargs.update({
        "augment": True,
        "mosaic": 1.0,
        "mixup": 0.5,
        "hsv_h": 0.02, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 45.0, "translate": 0.2, "scale": 0.8, "shear": 0.2,
        "fliplr": 0.5, "flipud": 0.5,
        "lr0": 0.01, "lrf": 0.01,
        "momentum": 0.937, "weight_decay": 0.0005
    })
    yolo_model.train(**train_kwargs)
    metrics = yolo_model.val(data=dataset_yaml, imgsz=imgsz)
    results_dict = {}
    try:
        if hasattr(metrics, "results_dict") and isinstance(metrics.results_dict, dict):
            results_dict.update(metrics.results_dict)
    except Exception:
        pass
    try:
        if hasattr(metrics, "kpts"):
            results_dict.setdefault("kpt/mAP50", float(metrics.kpts.map50))
            results_dict.setdefault("kpt/mAP", float(metrics.kpts.map))
    except Exception:
        pass
    results_dict.update({
        "project_dir": train_kwargs["project"],
        "run_name": train_kwargs["name"],
        "time_minutes": round((time.time() - start) / 60.0, 3)
    })
    try:
        best_path = yolo_model.ckpt_path if hasattr(yolo_model, "ckpt_path") else None
        if best_path and os.path.exists(best_path):
            results_dict["best_weights"] = best_path
    except Exception:
        pass
    return results_dict

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="WECO pose finetune with aug+cos")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default="weco-pose-aug-cos")
    args = parser.parse_args()
    out = finetune(
        dataset_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model=args.model,
        project_dir=args.project,
        name=args.name,
    )
    print(json.dumps(out, indent=2))