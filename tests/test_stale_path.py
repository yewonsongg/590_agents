"""Exercise real SDK serialization and local tools with every model response mocked."""

import asyncio
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from agents590 import provider, registry
from agents590.contracts import RunContext
from agents590.yewons_stale_path import StalePathAgent, tools
from agents590.yewons_stale_path.diagnostics import RunDiagnostics


@dataclass
class MockModel:
    replies: list[httpx.Response] = field(default_factory=list)
    requests: list[dict[str, Any]] = field(default_factory=list)
    client: httpx.AsyncClient = field(init=False)

    def respond(
        self,
        *output: dict[str, object],
        status: str = "completed",
        usage: dict[str, object] | None = None,
    ) -> None:
        self.replies.append(
            httpx.Response(
                200,
                json={
                    "id": f"resp_{len(self.replies)}",
                    "object": "response",
                    "created_at": 0,
                    "status": status,
                    "output": list(output),
                    "usage": usage,
                },
            )
        )


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> MockModel:
    for name in list(os.environ):
        if name.startswith(("OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "default-deployment")
    stub = MockModel()

    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{provider.UNC_ENDPOINT}/openai/v1/responses"
        assert request.headers["authorization"] == "Bearer synthetic-test-key"
        stub.requests.append(json.loads(request.content))
        assert stub.replies, "Unexpected additional model request"
        return stub.replies.pop(0)

    stub.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(provider, "AsyncOpenAI", partial(AsyncOpenAI, http_client=stub.client))
    return stub


def call(arguments: str, *, name: str = "shell", call_id: str = "call_1") -> dict[str, object]:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def message(text: str = "Repaired and verified.") -> dict[str, object]:
    return {
        "type": "message",
        "id": "msg_final",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def python_command(script: Path) -> str:
    if os.name == "nt":
        executable = sys.executable.replace("'", "''")
        argument = str(script).replace("'", "''")
        return f"& '{executable}' '{argument}'; exit $LASTEXITCODE"
    return shlex.join([sys.executable, str(script)])


def test_agent_is_registered_without_running_it() -> None:
    assert "yewons_stale_path" in registry.agent_names()
    assert isinstance(registry.create_agent("yewons_stale_path"), StalePathAgent)


def test_mocked_loop_repairs_files_and_replays_reasoning(model: MockModel, tmp_path: Path) -> None:
    (tmp_path / "resources").mkdir()
    asset = '{"rates": {"EUR": 0.8}}'
    (tmp_path / "resources/rates.json").write_text(asset)
    (tmp_path / "config.json").write_text('{"rates_file": "data/rates.json"}')
    (tmp_path / "report.py").write_text('RESOURCE = "data/rates.json"\n')
    inspect = tmp_path / "inspect_project.py"
    inspect.write_text("from pathlib import Path\nprint(Path('config.json').read_text())\n")
    repair = tmp_path / "repair_project.py"
    repair.write_text(
        "from pathlib import Path\n"
        "for name in ('config.json', 'report.py'):\n"
        "    p = Path(name)\n"
        "    p.write_text(p.read_text().replace('data/rates.json', 'resources/rates.json'))\n"
        "print('updated both references')\n"
    )
    reasoning: dict[str, object] = {
        "id": "rs_1",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "mock-encrypted-reasoning",
    }
    model.respond(reasoning, call(json.dumps({"command": python_command(inspect)})))
    model.respond(call(json.dumps({"command": python_command(repair)}), call_id="call_2"))
    model.respond(message())

    result = asyncio.run(
        StalePathAgent().run(
            "Repair stale resource references.",
            RunContext(cwd=tmp_path, model="openai/override-deployment"),
        )
    )

    assert result == "Repaired and verified."
    assert json.loads((tmp_path / "config.json").read_text()) == {
        "rates_file": "resources/rates.json"
    }
    assert (tmp_path / "report.py").read_text() == 'RESOURCE = "resources/rates.json"\n'
    assert (tmp_path / "resources/rates.json").read_text() == asset
    assert not (tmp_path / "data").exists()
    assert len(model.requests) == 3
    for request in model.requests:
        assert request["model"] == "override-deployment"
        assert request["store"] is False
        assert request["background"] is False
        assert request["max_output_tokens"] == 1024
        assert request["parallel_tool_calls"] is False
        assert request["include"] == ["reasoning.encrypted_content"]
        assert request["tools"][0]["strict"] is True
    second_input = model.requests[1]["input"]
    assert second_input[2] == reasoning
    assert second_input[3]["type"] == "function_call"
    assert second_input[4]["call_id"] == "call_1"
    assert json.loads(second_input[4]["output"])["exit_code"] == 0
    assert "data/rates.json" in json.loads(second_input[4]["output"])["stdout"]
    third_input = model.requests[2]["input"]
    assert third_input[: len(second_input)] == second_input
    assert third_input[-1]["call_id"] == "call_2"
    assert json.loads(third_input[-1]["output"])["stdout"].strip() == "updated both references"
    assert model.client.is_closed


@pytest.mark.parametrize(
    ("tool_name", "arguments", "error"),
    [
        ("unknown", "{}", "Unknown tool"),
        ("shell", "invalid JSON", "JSON object"),
        ("shell", '{"command": 7}', "nonempty string"),
    ],
)
def test_bad_calls_return_errors_to_model(
    model: MockModel,
    tmp_path: Path,
    tool_name: str,
    arguments: str,
    error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model.respond(call(arguments, name=tool_name))
    model.respond(message("Could not execute that call."))
    assert asyncio.run(StalePathAgent().run("Test", RunContext(cwd=tmp_path))) == (
        "Could not execute that call."
    )
    output = json.loads(model.requests[1]["input"][-1]["output"])
    assert error in output["error"]
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    tool_end = next(event for event in events if event["event"] == "tool_end")
    assert tool_end["status"] == "failed"
    assert error in tool_end["error"]


def test_step_limit_closes_client(
    model: MockModel, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model.respond(call("{}"))
    agent = StalePathAgent()
    agent.max_steps = 1
    with pytest.raises(ValueError, match="1-step limit"):
        asyncio.run(agent.run("Test", RunContext(cwd=tmp_path)))
    assert len(model.requests) == 1
    assert model.client.is_closed
    summary = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert summary["event"] == "run_end"
    assert summary["status"] == "step_limit"
    assert summary["requests"] == 1
    assert summary["input_tokens"] is None


def test_run_timeout_cancels_tool(
    model: MockModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model.respond(call('{"command": "unused"}'))
    cancelled: list[bool] = []

    async def slow_tool(
        arguments: str, cwd: Path, *, max_output_bytes: int, stats: tools.ShellStats
    ) -> str:
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.append(True)
        return "{}"

    monkeypatch.setattr("agents590.yewons_stale_path.execute_shell", slow_tool)
    agent = StalePathAgent()
    agent.run_timeout_seconds = 0.1
    with pytest.raises(ValueError, match="run limit"):
        asyncio.run(agent.run("Test", RunContext(cwd=tmp_path)))
    assert cancelled == [True]
    assert model.client.is_closed
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    tool_end = next(event for event in events if event["event"] == "tool_end")
    assert tool_end["status"] == "cancelled"
    assert events[-1]["status"] == "timeout"


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_unfinished_responses_do_not_execute_commands(
    model: MockModel, tmp_path: Path, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unexpected_tool(
        arguments: str, cwd: Path, *, max_output_bytes: int, stats: tools.ShellStats
    ) -> str:
        pytest.fail("An unfinished model response must not execute commands.")

    monkeypatch.setattr("agents590.yewons_stale_path.execute_shell", unexpected_tool)
    model.respond(call('{"command": "unused"}'), status=status)
    with pytest.raises(ValueError, match=f"status={status}"):
        asyncio.run(StalePathAgent().run("Test", RunContext(cwd=tmp_path)))
    assert len(model.requests) == 1
    assert model.client.is_closed


def test_empty_response_is_an_error(model: MockModel, tmp_path: Path) -> None:
    model.respond()
    with pytest.raises(ValueError, match="neither a tool call nor a final answer"):
        asyncio.run(StalePathAgent().run("Test", RunContext(cwd=tmp_path)))


def test_api_error_does_not_expose_provider_details(
    model: MockModel, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model.replies.append(
        httpx.Response(401, json={"error": {"message": "synthetic-test-key is invalid"}})
    )
    with pytest.raises(ValueError, match="AuthenticationError") as error:
        asyncio.run(StalePathAgent().run("Test", RunContext(cwd=tmp_path)))
    assert "synthetic-test-key" not in str(error.value)
    assert model.client.is_closed
    logs = capsys.readouterr().err
    assert "synthetic-test-key" not in logs
    summary = json.loads(logs.splitlines()[-1])
    assert summary["status"] == "api_error"
    assert summary["requests"] == 1
    assert summary["responses"] == 0


@pytest.mark.parametrize(
    "arguments",
    [
        "not json",
        "null",
        "[]",
        "{}",
        '{"command": "", "extra": true}',
        '{"command": ""}',
        '{"command": "  "}',
        '{"command": false}',
        '{"command": "\\u0000"}',
    ],
)
def test_shell_rejects_invalid_arguments(tmp_path: Path, arguments: str) -> None:
    assert "error" in json.loads(asyncio.run(tools.execute_shell(arguments, tmp_path)))


def test_shell_returns_failure_and_bounded_output(tmp_path: Path) -> None:
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\n"
        "print('OUT_START' + 'x' * 50000 + 'OUT_END')\n"
        "print('ERR_START' + 'e' * 50000 + 'failed check', file=sys.stderr)\n"
        "raise SystemExit(7)\n"
    )
    stats = tools.ShellStats()
    output = json.loads(
        asyncio.run(
            tools.execute_shell(
                json.dumps({"command": python_command(script)}), tmp_path, stats=stats
            )
        )
    )
    assert output["exit_code"] == 7
    assert output["stdout"].startswith("OUT_START")
    assert output["stdout"].rstrip().endswith("OUT_END")
    assert output["stderr"].startswith("ERR_START")
    assert output["stderr"].rstrip().endswith("failed check")
    assert "failed check" in output["stderr"]
    assert "output truncated" in output["stdout"]
    assert len(output["stdout"]) < tools.MAX_OUTPUT_BYTES + 100
    assert len(output["stderr"]) < tools.MAX_OUTPUT_BYTES + 100
    assert stats.stdout.total_bytes > 50_000
    assert stats.stderr.total_bytes > 50_000
    assert stats.stdout.retained_bytes == tools.MAX_OUTPUT_BYTES
    assert stats.stderr.retained_bytes == tools.MAX_OUTPUT_BYTES
    assert set(output) == {"exit_code", "stdout", "stderr"}


@pytest.mark.parametrize("limit", [1, 7, 8])
@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 9000])
def test_output_reader_preserves_boundaries_and_omission_count(limit: int, size: int) -> None:
    data = (b"abcdefghijklmnopqrstuvwxyz" * 347)[:size]
    stats = tools.OutputStats()

    async def exercise() -> str:
        stream = asyncio.StreamReader()

        async def feed() -> None:
            for offset in range(0, len(data), 3):
                stream.feed_data(data[offset : offset + 3])
                await asyncio.sleep(0)
            stream.feed_eof()

        producer = asyncio.create_task(feed())
        try:
            return await tools._read_output(stream, limit, stats=stats)
        finally:
            await producer

    output = asyncio.run(exercise())
    assert stats.as_dict() == {
        "total_bytes": size,
        "retained_bytes": min(size, limit),
        "omitted_bytes": max(0, size - limit),
        "truncated": size > limit,
    }
    if size <= limit:
        assert output == data.decode()
    else:
        head_size = (limit + 1) // 2
        tail_size = limit // 2
        expected_tail = data[-tail_size:].decode() if tail_size else ""
        assert output == (
            data[:head_size].decode()
            + f"\n[output truncated: {size - limit} bytes omitted]\n"
            + expected_tail
        )


def test_agent_uses_custom_output_limit(
    model: MockModel, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "output.py"
    script.write_text("print('START' + 'x' * 100 + 'END')\n")
    model.respond(call(json.dumps({"command": python_command(script)})))
    model.respond(message("Checked output."))
    agent = StalePathAgent()
    agent.max_shell_output_bytes = 16

    assert asyncio.run(agent.run("Test", RunContext(cwd=tmp_path))) == "Checked output."

    assert "16 bytes" in model.requests[0]["tools"][0]["description"]
    output = json.loads(model.requests[1]["input"][-1]["output"])
    assert output["stdout"].startswith("STARTxxx")
    assert output["stdout"].rstrip().endswith("END")
    expected_bytes = len(("START" + "x" * 100 + "END" + os.linesep).encode())
    assert f"{expected_bytes - 16} bytes omitted" in output["stdout"]
    assert output["stderr"] == ""
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    tool_end = next(event for event in events if event["event"] == "tool_end")
    assert tool_end["stdout"] == {
        "total_bytes": expected_bytes,
        "retained_bytes": 16,
        "omitted_bytes": expected_bytes - 16,
        "truncated": True,
    }
    assert tool_end["stderr"]["truncated"] is False
    assert tool_end["result_bytes"] == len(model.requests[1]["input"][-1]["output"].encode())


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_output_limit_fails_before_running_a_command(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="positive number of bytes"):
        tools.shell_tool(limit)
    with pytest.raises(ValueError, match="positive number of bytes"):
        asyncio.run(tools.execute_shell('{"command": "unused"}', tmp_path, max_output_bytes=limit))


def test_command_timeout_stops_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "slow.py"
    script.write_text(
        "import time\nfrom pathlib import Path\n"
        "time.sleep(2)\nPath('late.txt').write_text('orphaned process')\n"
    )
    monkeypatch.setattr(tools, "COMMAND_TIMEOUT_SECONDS", 1.0)
    stats = tools.ShellStats()

    async def exercise() -> dict[str, object]:
        output = json.loads(
            await tools.execute_shell(
                json.dumps({"command": python_command(script)}), tmp_path, stats=stats
            )
        )
        await asyncio.sleep(2)
        return output

    output = asyncio.run(exercise())
    assert "time limit" in str(output["error"])
    assert not (tmp_path / "late.txt").exists()
    assert stats.timed_out is True


def test_diagnostic_events_include_usage_commands_and_failures(
    model: MockModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def usage(input_tokens: int, output_tokens: int, cached_tokens: int) -> dict[str, object]:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 3},
        }

    command = "git diff # synthetic-test-key"
    model.respond(call(json.dumps({"command": command})), usage=usage(100, 20, 10))
    model.respond(message("Git unavailable; check files directly."), usage=usage(150, 30, 50))

    async def failed_tool(
        arguments: str, cwd: Path, *, max_output_bytes: int, stats: tools.ShellStats
    ) -> str:
        assert json.loads(arguments)["command"] == command
        error_output = "git: not found"
        stats.stderr.total_bytes = len(error_output)
        stats.stderr.retained_bytes = len(error_output)
        return json.dumps({"exit_code": 127, "stdout": "", "stderr": error_output})

    monkeypatch.setattr("agents590.yewons_stale_path.execute_shell", failed_tool)
    assert asyncio.run(StalePathAgent().run("Test", RunContext(cwd=tmp_path))) == (
        "Git unavailable; check files directly."
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "synthetic-test-key" not in captured.err
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert [event["event"] for event in events] == [
        "run_start",
        "model_request",
        "model_response",
        "tool_start",
        "tool_end",
        "model_request",
        "model_response",
        "run_end",
    ]
    assert events[2]["input_tokens"] == 100
    assert events[2]["cached_tokens"] == 10
    assert events[2]["reasoning_tokens"] == 3
    assert events[3]["command"] == "git diff # [REDACTED]"
    assert events[3]["call_id"] == events[4]["call_id"] == "call_1"
    assert events[3]["step"] == events[4]["step"] == 1
    assert events[4]["exit_code"] == 127
    assert events[4]["status"] == "failed"
    assert events[4]["stderr_excerpt"] == "git: not found"
    assert events[4]["duration_ms"] >= 0
    assert events[5]["history_items"] > events[1]["history_items"]
    assert events[-1]["status"] == "completed"
    assert events[-1]["requests"] == events[-1]["responses"] == 2
    assert events[-1]["tool_calls"] == 1
    assert events[-1]["input_tokens"] == 250
    assert events[-1]["output_tokens"] == 50
    assert events[-1]["cached_tokens"] == 60
    # Diagnostics never become extra context sent to the model.
    assert set(json.loads(model.requests[1]["input"][-1]["output"])) == {
        "exit_code",
        "stdout",
        "stderr",
    }


def test_diagnostic_text_is_bounded_and_redacted_before_clipping(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = 'synthetic-"key\\value'
    log = RunDiagnostics((secret,))
    command = "a" * 2040 + secret + "b" * 5000
    log.emit("tool_start", command=log.text(command), error=secret)
    captured = capsys.readouterr()
    event = json.loads(captured.err)
    assert "synthetic" not in captured.err
    assert event["error"] == "[REDACTED]"
    assert "[log text truncated]" in event["command"]
    assert len(event["command"]) < 4200


def test_missing_working_directory_is_a_tool_error(tmp_path: Path) -> None:
    output = json.loads(
        asyncio.run(tools.execute_shell('{"command": "unused"}', tmp_path / "missing"))
    )
    assert "Could not start shell" in output["error"]
