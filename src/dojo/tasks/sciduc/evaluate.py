from pathlib import Path
import importlib.util
import json
from datetime import datetime
import shutil
from sciduc.registry import Task, Registry, registry, TaskReport
from sciduc.utils import load_answers
import logging
logger = logging.getLogger(__name__)

# def evaluate_program(program_path: Path, private_dir: Path, results_output_dir: Path):
#     """
#         Evaluate the provided program using the evaluator in the private directory.
#         The evaluator should return a dictionary of metrics.
#         The program should be copied to `results_output_dir/programs/` w/ name based on the timestamp.
#         The following information should be appended to a list of reports in the grading report in the results output directory:
#             - metrics
#             - path_to_program
#             - private_dir 
#             - created_at        
#     """
#     # Load the evaluator
#     evaluator_spec = importlib.util.spec_from_file_location("evaluator", private_dir / "evaluator.py")
#     evaluator_module = importlib.util.module_from_spec(evaluator_spec)
#     evaluator_spec.loader.exec_module(evaluator_module)

#     if not hasattr(evaluator_module, "evaluate"):
#         raise ValueError("Evaluator module does not have an evaluate function")

#     # Evaluate the program
#     metrics = evaluator_module.evaluate(program_path, private_dir, results_output_dir)
#     program_name = f"program_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
#     program_copy_path = results_output_dir / "programs" / program_name
#     shutil.copy(program_path, program_copy_path)
#     save_path = results_output_dir / "grading_report.json"
#     report = {
#         "metrics": metrics.to_dict(),
#         "program_path": str(program_copy_path),
#         "private_dir": str(private_dir),
#         "results_output_dir": str(results_output_dir),
#         "created_at": datetime.now().isoformat(),
#     }
#     # Load the existing grading report
#     with open(save_path, "r") as f:
#         existing_report = json.load(f)  
#     # Append the report to the grading report
#     if "reports" not in existing_report:
#         existing_report["reports"] = []
#     with open(save_path, "w") as f:
#         existing_report["reports"].append(report)  
#         json.dump(existing_report, f, indent=2)
#     return metrics

def evaluate_submission(submission_path: Path, task_id: Path, results_output_dir: Path):
    new_registry = registry.set_data_dir(data_dir)
    task = new_registry.get_task(task_id)

    # if not is_dataset_prepared(task, grading_only=True):
        
    score = None
    submission_exists = submission_path.is_file() and submission_path.suffix.lower() == ".json"

    if submission_exists:
        with open(submission_path, "r") as f:
            submission = json.load(f)
        answers = task.answers  
        score = task.grader(submission, answers)
    else:
        logger.warning(
            f"Invalid submission file: {submission_path}. Please check that the file exists and it is a JSON."
        )
    valid_submission = score is not None
    report = TaskReport(
        task_id=task.id,
        score=score,
        valid_submission=valid_submission,
        created_at=datetime.now(),
        submission_path=str(submission_path),
    )
    results_output_dir.mkdir(exist_ok=True)
    save_path = results_output_dir / f"grading_report.json"
    with open(save_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    score = report.score
    score = float(score) if score is not None else None

    return score, report.to_dict()