import os
from pathlib import Path
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging
import shutil
logger = logging.getLogger(__name__)
import traceback

def evaluate_submission(submission_path: Path, annot_dir: Path, results_output_dir: Path):
    """
    Safely evaluate a submission file against ground-truth annotations.

    Returns:
        Tuple[float, dict]: (fitness_score, report_dict)
        - If evaluation fails, returns (-1.0, {"error": <message>}).
    """
    results_output_dir.mkdir(exist_ok=True, parents=True)
    debug_dir = results_output_dir / "debug"
    debug_dir.mkdir(exist_ok=True, parents=True)
    # Default fallbacks
    score = -1.0
    score_dict = {"error": "Unknown error: evaluation did not run."}

    # --- save a copy of the submission file (even if invalid) ---
    if submission_path.exists():
        debug_copy_path = debug_dir / f"submission_{submission_path.name}"
        try:
            shutil.copy2(submission_path, debug_copy_path)
            logger.info(f"Saved debug copy of submission to: {debug_copy_path}")
        except Exception as e:
            logger.warning(f"Failed to copy submission file for debugging: {e}")
    else:
        logger.warning(f"Submission file not found: {submission_path}")
        
    # Check submission file
    if not submission_path.is_file() or submission_path.suffix.lower() != ".json":
        msg = f"Invalid submission file: {submission_path}. Expected a .json file."
        logger.warning(msg)
        score_dict = {"error": msg}
        _save_report(results_output_dir, score_dict)
        return score, score_dict
    try:
        # Load ground truth and detections
        coco_gt = COCO(os.path.join(annot_dir, "val_annotations.json"))
        coco_dt = coco_gt.loadRes(str(submission_path))

        # Evaluate using COCOEval
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract key metrics (adjust depending on what you need)
        ap50 = float(coco_eval.stats[1])  # AP@0.5
        score_dict = {"AP@0.5": ap50, "all_stats": coco_eval.stats.tolist()}
        score = ap50

    except Exception as e:
        full_tb = traceback.format_exc()
        msg = f"Error evaluating submission: {e}\n{full_tb}"
        logger.warning(msg)
        score_dict = {"error": msg}
        score = -1.0


    _save_report(results_output_dir, score_dict)
    return score, score_dict


def _save_report(results_output_dir: Path, report: dict):
    """Helper: save evaluation results to grading_report.json."""
    try:
        save_path = results_output_dir / "grading_report.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved grading report to {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save grading report: {e}")


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


#evaluate_submission("", "/share/j_sun/ethan/sciduc/wildfin/private", ".")
if __name__ == "__main__":
    evaluate_submission(Path("/home/eyl45/Sun/aira-dojo/submission.json"), 
                        Path("/share/j_sun/ethan/sciduc/wildfin/private"),
                        Path("/home/eyl45/Sun/aira-dojo/"))