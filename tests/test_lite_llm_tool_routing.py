# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Ensures LiteLLMClient uses Chat Completions ``tools`` + correct ``tool_choice``
# for OpenAI vs Anthropic (no live API calls).

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from dojo.core.solvers.llm_helpers.backends.lite_llm import LiteLLMClient
from dojo.core.solvers.operators.analyze import analyze_schema_with_eval


def _fake_completion_with_tool_args(**kwargs):
    """Return a minimal completion object with one matching tool_call."""
    fn = SimpleNamespace(
        name="submit_review",
        arguments='{"is_bug": false, "summary": "ok", "metric": 0.5}',
    )
    tc = SimpleNamespace(function=fn)
    msg = SimpleNamespace(content=None, tool_calls=[tc], function_call=None)
    choice = SimpleNamespace(message=msg)
    comp = SimpleNamespace(choices=[choice], to_dict=lambda: {"usage": {}})
    return comp


@pytest.fixture
def openai_client():
    cfg = OmegaConf.load("src/dojo/configs/solver/client/litellm_4o.yaml")
    return LiteLLMClient(instantiate(cfg))


@pytest.fixture
def anthropic_client():
    cfg = OmegaConf.load("src/dojo/configs/solver/client/litellm_claude.yaml")
    return LiteLLMClient(instantiate(cfg))


def test_openai_route_uses_tools_and_tool_choice_auto(openai_client):
    captured: dict = {}

    def capture(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_completion_with_tool_args(**kwargs)

    assert not openai_client._is_anthropic_route()

    with patch("dojo.core.solvers.llm_helpers.backends.lite_llm.completion_fn", side_effect=capture):
        out, _ = openai_client.query(
            [{"role": "user", "content": "hi"}],
            json_schema=analyze_schema_with_eval,
            function_name="submit_review",
            function_description="review",
        )

    assert captured.get("tool_choice") == "auto"
    assert "tools" in captured
    assert "functions" not in captured
    tool0 = captured["tools"][0]
    assert tool0["type"] == "function"
    assert tool0["function"]["name"] == "submit_review"
    assert "parameters" in tool0["function"]
    assert isinstance(out, dict)
    assert out.get("metric") == 0.5


def test_anthropic_route_forces_tool_choice(anthropic_client):
    captured: dict = {}

    def capture(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return _fake_completion_with_tool_args(**kwargs)

    assert anthropic_client._is_anthropic_route()

    with patch("dojo.core.solvers.llm_helpers.backends.lite_llm.completion_fn", side_effect=capture):
        anthropic_client.query(
            [{"role": "user", "content": "hi"}],
            json_schema=analyze_schema_with_eval,
            function_name="submit_review",
            function_description="review",
        )

    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_review"},
    }
    assert "tools" in captured and "functions" not in captured
