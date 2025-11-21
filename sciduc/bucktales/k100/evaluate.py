"""
Evaluator for object detection programs using the new ObjectDetectionDataset.
"""
import sys
import os
# Add the Apptainer bind mount path to Python's module search
sys.path.insert(0, "/work")

print("✅ Added /work to sys.path")
print("Working dir:", os.getcwd())

import time
import json
import logging
import importlib.util
from pycocotools.coco import COCO
from src.eval import evaluate as evaluate_coco
from src.dataset import ObjectDetectionDataset
import traceback
import ultralytics

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def evaluate(program_path, data_dir, ann_dir, seed=42):
    """
    Main evaluation function that tests the object detection method
    on the validation set and computes the composite performance metric.

    Args:
        program_path (str): Path to user program (must implement finetune()).
        data_dir (str): Path to image root directory (contains train/, val/, test/).
        ann_dir (str): Path to annotation directory (contains train.json, val.json, etc.).
        seed (int): Random seed.
    """
    start_time = time.time()

    # Dynamically import the user program
    logging.info(f"Loading program from {program_path}")
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    if not hasattr(program, "finetune"):
        raise AttributeError("❌ The program must define a 'finetune(train_dataset, val_dataset)' function.")

    # Load datasets
    logging.info("Loading COCO-format datasets...")
    train_dataset = ObjectDetectionDataset(
        data_dir=data_dir,
        ann_dir=ann_dir,
        split="train",
    )
    val_dataset = ObjectDetectionDataset(
        data_dir=data_dir,
        ann_dir=ann_dir,
        split="val",
    )

    # Run the user program
    logging.info("Starting finetune()...")
    try:
        coco_results = program.finetune(train_dataset, val_dataset)
    except Exception as e:
        logging.error("❌ Error during finetune() execution:")
        logging.error(traceback.format_exc())
        return {"error": str(e)}

    if not coco_results:
        return {"error": "No detection results returned by finetune()."}

    # Load COCO ground truth and detection results
    ann_file = os.path.join(ann_dir, "val", "annotations.json")
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(coco_results)

    # Evaluate using COCO metrics
    logging.info("Evaluating predictions...")
    total_metrics = evaluate_coco(coco_gt, coco_dt)
    total_metrics["time_minutes"] = round((time.time() - start_time) / 60, 3)
    total_metrics["program_path"] = program_path

    logging.info("✅ Evaluation complete.")
    return total_metrics


if __name__ == "__main__":
    # Default paths are Apptainer-friendly
    ultralytics.utils.LOGGER.setLevel("ERROR")  # only errors will print
    root_dir = os.environ.get("ROOT_DIR", "/work")
    program_path = os.environ.get("PROGRAM_PATH", os.path.join(root_dir, "program.py"))
    data_dir = os.environ.get("DATA_DIR", os.path.join(root_dir, "/data"))
    ann_dir = os.environ.get("ANN_DIR", os.path.join(root_dir, "annotations"))

    logging.info(f"ROOT_DIR={root_dir}")
    logging.info(f"DATA_DIR={data_dir}")
    logging.info(f"ANN_DIR={ann_dir}")

    metrics = evaluate(program_path, data_dir, ann_dir)
    metrics["fitness"] = metrics.get("AP@0.5", 0.0)
    print("METRICS:", json.dumps(metrics, indent=2))
