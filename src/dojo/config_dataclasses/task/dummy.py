from dataclasses import dataclass, field
from dojo.config_dataclasses.task.base import TaskConfig


@dataclass
class DummyConfig(TaskConfig):
    benchmark: str = field(
        default="dummy",
        metadata={
            "help": "Type of the task.",
        },
    )

    submission_fname: str = field(
        default="submission.json",
        metadata={
            "help": "Name of the submission file.",
        },
    )

    def validate(self) -> None:
        super().validate()