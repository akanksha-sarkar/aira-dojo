# dataclasses is a built-in Python module for creating data classes
from dataclasses import dataclass, field

# omegaconf is a YAML configuration system for Python
# SI = Structured Interpolation, MISSING = sentinel value for required fields
from omegaconf import SI, MISSING

from dojo.config_dataclasses.task.base import TaskConfig
from dojo.utils.environment import get_sciduc_data_dir

sciduc_operator_prompts: dict[str, str] = {
    "draft_intro": """TBD""",
    "improve_intro": """TBD""",
    "debug_intro": """TBD""",
    "analyze_intro": """TBD""",
    "implementation_guideline": """TBD""",
    "execution_intro": """TBD""",
    "response_format": """TBD""",
    "solution_sketch": """TBD""",
    "improvement_sketch": """TBD""",
    "bugfix_sketch": """TBD""",
    "analyze_func_name": """TBD""",
    "analyze_func_desc": """TBD""",
    "crossover_intro": """TBD""",
    "crossover_sketch": """TBD""",
}

@dataclass
class SciDUCTaskConfig(TaskConfig):
    benchmark: str = field(
        default="sciduc",
        metadata={
            "help": "Type of the task.",
        },
    )
    data_dir: str = field(
        default=SI("${task.public_dir}"),
        metadata={
            "help": "The directory where the data is stored.",
            "exclude_from_hash": True,
        },
    )
    results_output_dir: str = field(
        default=SI("${logger.output_dir}/results"),
        metadata={
            "help": "The directory where the results are stored.",
            "exclude_from_hash": True,
        },
    )
    def validate(self) -> None:
        super().validate()