#!/usr/bin/env python3
"""
One-shot check that LiteLLM + analyze-style tool calling works (incl. forced tool_choice on Claude).

Run from repo root:
  PYTHONPATH=src python scripts/smoke_analyze_tool_choice.py

Requires ANTHROPIC_API_KEY or PRIMARY_KEY (see litellm_claude.yaml / LiteLLMClient).
LiteLLMClient sets drop_params for Anthropic; if you test litellm.completion() directly, use
drop_params=True so OpenAI-style `functions` is translated for Claude.
"""
from __future__ import annotations

import sys
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient
from dojo.core.solvers.operators.analyze import analyze_schema_with_eval


def main() -> None:
    cfg_path = _REPO / "src/dojo/configs/solver/client/litellm_claude.yaml"
    if not cfg_path.is_file():
        print(f"Missing config: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    client = LiteLLMClient(instantiate(OmegaConf.load(cfg_path)))

    messages = [
        {
            "role": "system",
            "content": (
                "You evaluate a training run.\n"
                "Execution output:\n```\nMETRICS: {'fitness': 0.42}\n```\n"
                "Call submit_review with is_bug, summary, and metric."
            ),
        },
        {
            "role": "user",
            "content": "Follow the system instructions and respond with the requested output.",
        },
    ]

    out, usage = client.query(
        messages,
        json_schema=analyze_schema_with_eval,
        function_name="submit_review",
        function_description="Submit a review evaluating the output of the training script.",
    )

    print("success=True")
    print("output type:", type(out).__name__)
    print("output:", out)
    print("usage keys:", sorted(usage.keys()))


if __name__ == "__main__":
    main()
