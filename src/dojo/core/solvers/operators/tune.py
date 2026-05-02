import os
import random
import time
from typing import Callable, Optional

import humanize
from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.journal import Journal, Node
from dojo.core.solvers.utils.response import wrap_code


def tune_op(
    tune_llm: GenericLLM,
    cfg: DictConfig,
    memory_op: Optional[Callable[[Journal, Optional[Node]], str]],
    task_description: str,
    journal: Journal,
    input_node: Node,
    step_count: int,
    remaining_time: int,
    data_preview: Optional[str] = None,
) -> str:
    pkgs = cfg.available_packages
    random.shuffle(pkgs)
    pkg_str = ", ".join([f"`{p}`" for p in pkgs])

    if memory_op is not None:
        memory = memory_op(journal, input_node)
    else:
        memory = ""

    exec_timeout = int(min(cfg.execution_timeout, remaining_time))
    steps_remaining = cfg.step_limit - step_count

    tune_data = {
        "task_desc": task_description,
        "prev_code": wrap_code(input_node.code),
        "prev_terminal_output": wrap_code(input_node.term_out, lang=""),
        "time_remaining": humanize.naturaldelta(remaining_time),
        "steps_remaining": steps_remaining,
        "execution_timeout": humanize.naturaldelta(exec_timeout),
        "other_remarks": None,
        "packages": pkg_str,
        "memory": None,
        "data_overview": None,
    }

    if memory:
        tune_data["memory"] = memory

    if cfg.data_preview and data_preview is not None:
        tune_data["data_overview"] = data_preview
    else:
        tune_data["data_overview"] = "(No data preview available)"

    return tune_llm(query_data=tune_data, no_user_message=True)
