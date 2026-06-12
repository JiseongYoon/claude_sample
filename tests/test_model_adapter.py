"""Tests for the LLMToolModel adapter.

Parses OpenAI chat-completions dicts into AssistantTurns, robustly. Run in conda
`local-ai-agent-env-1`: `pytest`.
"""
from __future__ import annotations

import pytest

from local_ai_agent.modules.orchestrator.model_adapter import LLMToolModel, parse_turn


class FakeChat:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.seen: list[tuple[list[dict], dict]] = []

    async def chat(self, messages, **params):
        self.seen.append((messages, params))
        return self.response


def _msg_text(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def _msg_tools(calls):
    return {"choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": calls}}]}


def _call(cid, name, arguments):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


# --- parse_turn: NORMAL ---------------------------------------------------- #
def test_text_only_turn():
    turn = parse_turn(_msg_text("final answer"))
    assert turn.text == "final answer"
    assert turn.tool_calls == []


def test_tool_calls_parsed():
    turn = parse_turn(_msg_tools([
        _call("a1", "read_file", '{"path": "workspace/x"}'),
        _call("a2", "delete_file", '{"path": "workspace/y"}'),
    ]))
    assert [c.tool for c in turn.tool_calls] == ["read_file", "delete_file"]
    assert turn.tool_calls[0].args == {"path": "workspace/x"}
    assert turn.tool_calls[0].id == "a1"


def test_arguments_as_dict_passthrough():
    turn = parse_turn(_msg_tools([_call("a1", "read_file", {"path": "x"})]))
    assert turn.tool_calls[0].args == {"path": "x"}


# --- parse_turn: ERROR / robustness ---------------------------------------- #
def test_unparseable_arguments_become_empty():
    turn = parse_turn(_msg_tools([_call("a1", "shell", "this is not json {{{")]))
    assert turn.tool_calls[0].args == {} # safe — gate still classifies "shell"


def test_non_dict_json_arguments_become_empty():
    turn = parse_turn(_msg_tools([_call("a1", "read_file", "[1,2,3]")]))
    assert turn.tool_calls[0].args == {}


def test_missing_choices_is_safe():
    assert parse_turn({}).text == ""
    assert parse_turn({"choices": []}).text == ""


@pytest.mark.parametrize("bad", ["hello", 123, [1, 2], None])
def test_non_dict_message_is_safe(bad):
    # round-1 defect: a non-dict message must not crash parse_turn
    turn = parse_turn({"choices": [{"message": bad}]})
    assert turn.text == "" and turn.tool_calls == []


@pytest.mark.parametrize("bad", ["notalist", 5, {"x": 1}])
def test_non_list_tool_calls_is_safe(bad):
    turn = parse_turn({"choices": [{"message": {"content": "t", "tool_calls": bad}}]})
    assert turn.tool_calls == [] and turn.text == "t"


def test_tool_call_without_name_skipped():
    turn = parse_turn(_msg_tools([{"id": "x", "function": {"arguments": "{}"}}]))
    assert turn.tool_calls == []


@pytest.mark.parametrize("bad_fn", ["abc", 5, [1, 2], True, 3.14])
def test_non_dict_function_skipped(bad_fn):
    # round-2 defect: a truthy non-dict `function` must not crash parse_turn
    turn = parse_turn(_msg_tools([{"id": "c1", "function": bad_fn}]))
    assert turn.tool_calls == []


@pytest.mark.parametrize("bad_choices", [{"x": 1}, {"0": "v"}, "str", 5, True, {}])
def test_non_list_choices_is_safe(bad_choices):
    # round-3 defect: a truthy non-list `choices` (e.g. a dict) must not crash parse_turn
    turn = parse_turn({"choices": bad_choices})
    assert turn.text == "" and turn.tool_calls == []


def test_parse_turn_never_raises_on_arbitrary_junk():
    for junk in [None, 5, "x", [], {"choices": [None]}, {"choices": [{"message": {"tool_calls": [None, 1, "a"]}}]}]:
        assert parse_turn(junk).tool_calls == [] # never raises


def test_missing_id_gets_synthetic():
    turn = parse_turn(_msg_tools([{"function": {"name": "read_file", "arguments": "{}"}}]))
    assert turn.tool_calls[0].id == "call-0"


# --- LLMToolModel.complete ------------------------------------------------- #
async def test_complete_passes_tools_and_parses():
    chat = FakeChat(_msg_text("done"))
    model = LLMToolModel(chat, temperature=0.0)
    turn = await model.complete([{"role": "user", "content": "hi"}], tools=[{"name": "t"}])
    assert turn.text == "done"
    msgs, params = chat.seen[0]
    assert params["tools"] == [{"name": "t"}]
    assert params["temperature"] == 0.0


async def test_complete_no_tools_omits_key():
    chat = FakeChat(_msg_text("x"))
    model = LLMToolModel(chat)
    await model.complete([{"role": "user", "content": "hi"}], tools=[])
    _, params = chat.seen[0]
    assert "tools" not in params


async def test_complete_propagates_engine_error():
    class Boom:
        async def chat(self, messages, **params):
            raise RuntimeError("engine not ready")

    with pytest.raises(RuntimeError):
        await LLMToolModel(Boom()).complete([], tools=[])
