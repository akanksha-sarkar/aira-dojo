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
    return results


if __name__ == "__main__":
    # Environment-variable friendly defaults
    # PROGRAM_PATH: path to the user's program file implementing `finetune`
    # DATASET_YAML: path to Ultralytics pose YAML
    program_path = os.environ.get("PROGRAM_PATH", os.path.join(os.getcwd(), "pose_estimation.py"))
    dataset_yaml = os.environ.get("DATASET_YAML", os.environ.get("POSE_DATA_YAML", ""))

    # Allow CLI overrides
    import argparse

    parser = argparse.ArgumentParser(description="WECO pose evaluation")
    parser.add_argument("--program", type=str, default=program_path)
    parser.add_argument("--data", type=str, default=dataset_yaml)
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 20)))
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("IMGSZ", 640)))
    args = parser.parse_args()

    metrics = evaluate(args.program, args.data, epochs=args.epochs, imgsz=args.imgsz)
    print("METRICS:", json.dumps(metrics, indent=2))


