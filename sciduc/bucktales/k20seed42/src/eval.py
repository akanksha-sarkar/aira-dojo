# from faster_coco_eval import COCOeval_faster
from pycocotools.cocoeval import COCOeval

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
    return metrics
