# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dojo.core.interpreters.base import ExecutionResult, Interpreter
from dojo.core.tasks.base import Task
from dojo.core.tasks.constants import (
    EXECUTION_OUTPUT,
    TASK_DESCRIPTION,
    TEST_FITNESS,
    VALID_SOLUTION_FEEDBACK,
    VALIDATION_FITNESS,
    AUX_EVAL_INFO,
    VALID_SOLUTION,
)
from dojo.utils.code_parsing import extract_code

from dojo.config_dataclasses.task.detection import DetectionConfig
from dojo.tasks.detection.evaluate import evaluate_submission



def validate_submission(submission: Path) -> tuple[bool, str]:
    """
    Validates a submission for the given competition by actually running the competition grader.
    This is designed for end users, not developers (we assume that the competition grader is
    correctly implemented and use that for validating the submission, not the other way around).
    """
    if not submission.is_file():
        return False, f"Submission invalid! Submission file {submission} does not exist."

    if not submission.suffix.lower() == ".json":
        return False, "Submission invalid! Submission file must be a JSON file."

    return True, "Submission is valid."

def parse_report(report: Dict[str, Any]):
    parsed_report = {}
    for key, value in report.items():
        if isinstance(value, (bool, int, float)):
            parsed_report[key] = float(value)
        else:
            parsed_report[key] = value
    return parsed_report


class DetectionTask(Task):
    """
    Represents an Wildfin task.

    This task executes a provided solution to train a model and expects the solution
    to produce a submission file (submission.csv). It then runs an evaluation script to
    compute a fitness score.

    Expected configuration keys (accessible via self.cfg):
      - eval_script: The evaluation script code.
      - task_description: The task description.
    """

    _solution_script = "solution.py"
    _submission_file_path = None

    def __init__(self, cfg: DetectionConfig) -> None:
        """
        Initialize the WildfinTask.

        Args:
            **cfg: Configuration containing keys like 'eval_script' and 'task_description'.
        """
        super().__init__(cfg)

        # Get instructions
        self.task_src_path = Path(__file__).resolve().parent
        self.instructions_path = self.task_src_path / "instructions.txt"
        self.instructions = self.instructions_path.read_text()
        self.instructions = os.path.expandvars(self.instructions)

        # Read task description.
        task_description_path = Path(self.cfg.public_dir).resolve() / "description.md"
        self.task_description = self.instructions + "\n" + task_description_path.read_text()

        # Resolve paths
        self.public_dir = Path(self.cfg.public_dir).resolve()
        self.private_dir = Path(self.cfg.private_dir).resolve()

    def prepare(self, **task_args):
        state = task_args
        state["init_obs"] = {}
        self._submission_file_path = Path(task_args["solver_interpreter"].working_dir) / self.cfg.submission_fname

        task_info = {
            TASK_DESCRIPTION: self.task_description,
            "lower_is_better": False,
        }

        return state, task_info

    def step_task(self, state: Dict[str, Any], action: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a single step of the task.

        In this task, a step corresponds to evaluating the provided solution code.

        Args:
            state (Dict[str, Any]): The current state of the task.
            action (Any): The solution code (as a string) to evaluate.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing the updated state and the outcome.
        """
        try:
            solution = extract_code(action)
        except Exception as e:
            self.logger.error(f"The solution does not follow the required format: {e}")
            exec_output = ExecutionResult.get_empty()
            exec_output.term_out[0] = f"Invalid solution: {e}"
            return state, {EXECUTION_OUTPUT: exec_output, VALIDATION_FITNESS: None, VALID_SOLUTION: False}

        interpreter = state["solver_interpreter"]
        exec_output: ExecutionResult = interpreter.run(solution, file_name=self._solution_script)
        eval_result = {EXECUTION_OUTPUT: exec_output}

        # Check if the execution was not successful
        if (not exec_output.exit_code == 0) or exec_output.timed_out:
            self.logger.error(
                f"Execution failed - exit code: {exec_output.exit_code} - timed out: {exec_output.timed_out} - execution time: {exec_output.exec_time}"
            )
            # If the execution was not succesful, for sanity reasons, we want to remove the submission file if it exists
            # this ensures that the agent does not have access to a submission file that was not generated by a successful execution
            if self._submission_file_path.exists():
                self._submission_file_path.unlink(missing_ok=True)
        else:
            self.logger.info(f"Execution successful - fetching submission file for evaluation.")
            interpreter.fetch_file(self._submission_file_path)
            self.logger.info(f"Submission file fetched: {self._submission_file_path}")

        has_json_submission = self._submission_file_path.exists()
        eval_result[VALID_SOLUTION] = False
        if has_json_submission:
            is_valid_submission, message = validate_submission(self._submission_file_path, self.cfg.name)
            eval_result[VALID_SOLUTION] = is_valid_submission
            eval_result[VALID_SOLUTION_FEEDBACK] = message
            self.logger.info(
                f"Submission file found: {self._submission_file_path} || Submission valid: {is_valid_submission}"
            )

            if is_valid_submission:
                self.logger.info(f"Evaluating submission: {self._submission_file_path}")
                test_fitness, report = evaluate_submission(
                    submission_path=self._submission_file_path,
                    data_dir=Path(self.cfg.cache_dir),
                    results_output_dir=Path(self.cfg.results_output_dir),
                )
                eval_result[TEST_FITNESS] = test_fitness
                eval_result[AUX_EVAL_INFO] = report
                self.logger.info(f"Test fitness: {test_fitness} || AUX eval info: {eval_result[AUX_EVAL_INFO]}")

            self._submission_file_path.unlink(missing_ok=True)  # remove the submission_file locally
            assert not self._submission_file_path.exists(), (
                "At this point, the submissions file should not exists locally!"
            )

        if interpreter.factory:
            interpreter.close()
        else:
            interpreter.run(
                f"!rm {self.cfg.submission_fname}"
            )  # remove the submission_file from the agent's environment
            assert interpreter.fetch_file(self._submission_file_path) is None, (
                "At this point, the submissions file should not exists in the agent's environment!"
            )

        return state, eval_result

    def evaluate_fitness(
        self,
        solution: Optional[Any] = None,
        state: Optional[Dict[str, Any]] = None,
        interpreter: Optional[Interpreter] = None,
        aux_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._submission_file_path is None:
            raise Exception("The path to the submission file must be set.")

        exec_output = interpreter.run(solution, file_name=self._solution_script)
        eval_result = {EXECUTION_OUTPUT: exec_output}

        interpreter.fetch_file(self._submission_file_path)
        has_json_submission = self._submission_file_path.exists()
        assert has_json_submission, "The final solution is not valid!"

        test_fitness, report = evaluate_submission(
            submission_path=self._submission_file_path,
            data_dir=Path(self.cfg.cache_dir),
            results_output_dir=Path(self.cfg.results_output_dir),
        )
        eval_result[TEST_FITNESS] = test_fitness
        eval_result[AUX_EVAL_INFO] = report

        return eval_result

    def close(self, state):
        for interp_key in ["solver_interpreter", "eval_interpreter"]:
            if interp_key not in state:
                continue

            interpreter = state[interp_key]
            if hasattr(interpreter, "cleanup_session"):
                interpreter.cleanup_session()

            if hasattr(interpreter, "clean_up"):
                interpreter.clean_up()
