import os
from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging
logger = logging.getLogger(__name__)

def evaluate_submission(submission_path: Path, data_dir: Path, results_output_dir: Path):


    score = None
    submission_exists = submission_path.is_file() and submission_path.suffix.lower() == ".json"

    if submission_exists:
        coco_gt = COCO(os.path.join(data_dir, "annotations.json")) # path to test annotations in the task folder
        coco_dt = coco_gt.loadRes(submission_path)
        score_dict = evaluate(coco_gt, coco_dt)
        logger.warning(
            f"Invalid submission file: {submission_path}. Please check that the file exists and it is a JSON."
            f"Score: {score_dict}"
        )

    valid_submission = score_dict is not None

    # report = ... (ignore report for now)

    # Ensure the output directory exists and write the report to disk.
    results_output_dir.mkdir(exist_ok=True)
    save_path = results_output_dir / f"grading_report.json"
    with open(save_path, "w") as f:
        json.dump( score_dict, f, indent=2) # ignore report for now

    score = score_dict["AP@0.5"]
    score = float(score) if score is not None else None
    return score, score_dict

def evaluate(coco_gt, coco_dt, verbose=False):
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    labels = [
        "AP@[.5:.95]",
        "AP@0.5",
        "AP@0.75",
        "AP (small)",
        "AP (medium)",
        "AP (large)",
        "AR@1",
        "AR@10",
        "AR@100",
        "AR (small)",
        "AR (medium)",
        "AR (large)"
    ]
    metrics = {}
    # print("Evaluation results:")
    for name, value in zip(labels, stats):
        # print(f"{name:<15}: {value:.4f}")
        metrics[name] = value
    return {"AP@0.5": metrics["AP@0.5"]}
