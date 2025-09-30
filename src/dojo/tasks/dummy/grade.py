from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def grade(submission, answer):
    """
      Answer is a path to a JSON file containing the ground truth annotations in COCO format.
      Submission is a path to a json file containing the predictions in COCO format.
    """
    coco_gt = COCO(answer)
    with open(submission, "r") as f:
        coco_predictions = json.load(f)
    coco_dt = coco_gt.loadRes(coco_predictions)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type="bbox")
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
    metrics = {"AP@0.5": metrics["AP@0.5"]}
    return metrics

def validate_submission(submission: Path, task: Task) -> tuple[bool, str]:
    """
    Validates a submission for the given competition by actually running the competition grader.
    This is designed for end users, not developers (we assume that the competition grader is
    correctly implemented and use that for validating the submission, not the other way around).
    """
    if not submission.is_file():
        return False, f"Submission invalid! Submission file {submission} does not exist."

    if not submission.suffix.lower() == ".json":
        return False, "Submission invalid! Submission file must be a CSV file."

    if not is_dataset_prepared(task, grading_only=True):
        raise ValueError("Dataset for task is not prepared!")

    try:
        grade(submission, task.answers)
    except Exception as e:
        return (
            False,
            f"Submission invalid! The attempt to grade the submission has resulted in the following error message:\n{e}",
        )

    return True, "Submission is valid."