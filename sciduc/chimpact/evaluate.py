"""
Evaluator for WECO pose estimation programs (Ultralytics-based).

Dynamically imports a user program module exposing `finetune(dataset_yaml, ...)`,
runs training/validation, and returns a compact metrics dict.
"""

import os
import sys
import json
import time
import logging
import importlib.util
import traceback
from typing import Dict


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def evaluate(program_path: str, dataset_yaml: str, epochs: int = 20, imgsz: int = 640) -> Dict:
    """
    Evaluate a pose estimation program by calling its `finetune` function.

    Args:
        program_path: Path to python module implementing `finetune(dataset_yaml, ...)`.
        dataset_yaml: Path to Ultralytics COCO-pose YAML config.
        epochs: Training epochs to run.
        imgsz: Image size for training/validation.

    Returns:
        Dict with metrics and run metadata.
    """
    start = time.time()
    
    # Debug logging - always show these when evaluate() is called
    root_dir = os.environ.get("ROOT_DIR", "/work")
    logging.info(f"=== Evaluation Starting ===")
    logging.info(f"ROOT_DIR: {root_dir}")
    logging.info(f"Program path: {program_path}")
    logging.info(f"Dataset YAML: {dataset_yaml}")
    
    # Debug: Check directory structure and YAML content
    if os.path.exists(dataset_yaml):
        logging.info(f"Reading dataset YAML: {dataset_yaml}")
        with open(dataset_yaml, 'r') as f:
            yaml_content = f.read()
            logging.info(f"YAML content:\n{yaml_content[:500]}...")  # First 500 chars
        
        # Check if images and labels directories exist
        images_train = os.path.join(root_dir, "images", "train")
        labels_train = os.path.join(root_dir, "labels", "train")
        logging.info(f"Images train dir exists: {os.path.exists(images_train)} - {images_train}")
        logging.info(f"Labels train dir exists: {os.path.exists(labels_train)} - {labels_train}")
        
        if os.path.exists(images_train):
            img_files = list(os.listdir(images_train))[:5]  # First 5
            logging.info(f"Sample image files: {img_files}")
        if os.path.exists(labels_train):
            label_files = list(os.listdir(labels_train))[:5]  # First 5
            logging.info(f"Sample label files: {label_files}")
            
            # Check if filenames match
            if os.path.exists(images_train) and os.path.exists(labels_train):
                img_bases = {os.path.splitext(f)[0] for f in os.listdir(images_train) if os.path.isfile(os.path.join(images_train, f))}
                label_bases = {os.path.splitext(f)[0] for f in os.listdir(labels_train) if os.path.isfile(os.path.join(labels_train, f))}
                matches = img_bases & label_bases
                logging.info(f"Image files: {len(img_bases)}, Label files: {len(label_bases)}, Matches: {len(matches)}")
                if len(matches) < len(img_bases):
                    missing = img_bases - label_bases
                    logging.warning(f"Images without labels: {len(missing)} (sample: {list(missing)[:5]})")
                    missing_labels = label_bases - img_bases
                    logging.warning(f"Labels without images: {len(missing_labels)} (sample: {list(missing_labels)[:5]})")

    if not os.path.exists(program_path):
        raise FileNotFoundError(f"Program not found: {program_path}")
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    logging.info(f"Loading program from {program_path}")
    spec = importlib.util.spec_from_file_location("program", program_path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)  # type: ignore

    if not hasattr(program, "finetune"):
        raise AttributeError("The program must define `finetune(dataset_yaml, ...)`.")

    logging.info("Starting finetune()...")
    try:
        results = program.finetune(dataset_yaml=dataset_yaml, epochs=epochs, imgsz=imgsz)
    except Exception as e:
        logging.error("Error during finetune() execution:")
        logging.error(traceback.format_exc())
        return {"error": str(e)}

    if not isinstance(results, dict):
        return {"error": "finetune() did not return a dict of metrics."}

    results.setdefault("time_minutes_total", round((time.time() - start) / 60.0, 3))
    logging.info("Evaluation complete.")
    logging.info(f"=== Evaluation Complete ===")
    return results


# if __name__ == "__main__":
#     # Environment-variable friendly defaults
#     # PROGRAM_PATH: path to the user's program file implementing `finetune`
#     # DATASET_YAML: path to Ultralytics pose YAML
#     # program_path = os.environ.get("PROGRAM_PATH", os.path.join(os.getcwd(), "pose_estimation.py"))
#     # dataset_yaml = os.environ.get("DATASET_YAML", os.environ.get("POSE_DATA_YAML", ""))

#     program_path = "/home/as2637/sciduc/aira-dojo/sciduc/chimpact/program.py"
#     dataset_yaml = "/share/j_sun/as2637/sciduc/chimpact/k100/dataset.yaml"

#     # Allow CLI overrides
#     import argparse

#     parser = argparse.ArgumentParser(description="WECO pose evaluation")
#     parser.add_argument("--program", type=str, default=program_path)
#     parser.add_argument("--data", type=str, default=dataset_yaml)
#     parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 20)))
#     parser.add_argument("--imgsz", type=int, default=int(os.environ.get("IMGSZ", 640)))
#     args = parser.parse_args()

#     metrics = evaluate(args.program, args.data, epochs=args.epochs, imgsz=args.imgsz)
#     print("METRICS:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    root_dir = os.environ.get("ROOT_DIR", "/work")
    logging.info(f"=== Evaluation Starting ===")
    logging.info(f"ROOT_DIR: {root_dir}")
    program_path = os.path.join(root_dir, "program.py")
    dataset_yaml = os.path.join(root_dir, "dataset.yaml")
    logging.info(f"Program path: {program_path}")
    logging.info(f"Dataset YAML: {dataset_yaml}")

    metrics = evaluate(program_path, dataset_yaml)
    logging.info(f"=== Evaluation Complete ===")
    print("METRICS:", json.dumps(metrics, indent=2))


