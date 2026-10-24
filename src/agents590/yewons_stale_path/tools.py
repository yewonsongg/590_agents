"""Validated shell execution with bounded output and process cleanup."""

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import cast

from openai.types.responses import FunctionToolParam

COMMAND_TIMEOUT_SECONDS = 10.0
MAX_OUTPUT_BYTES = 4096
SHELL_NAME = "PowerShell" if os.name == "nt" else "POSIX /bin/sh"


def shell_tool(max_output_bytes: int = MAX_OUTPUT_BYTES) -> FunctionToolParam:
    """Describe the shell with the same output limit used by its executor."""
    if max_output_bytes < 1:
        raise ValueError("Shell output limit must be a positive number of bytes.")
    return {
        "type": "function",
        "name": "shell",
        "description": (
            f"Run foreground {SHELL_NAME} commands in the task directory. Python is available. "
            "Each call starts there; directory changes do not persist. "
            f"Timeout: {COMMAND_TIMEOUT_SECONDS:g}s. Per stream: {max_output_bytes} bytes, "
            "preserving beginning/end and marking omissions. "
            "Returns JSON: exit_code, stdout, stderr, or error."
        ),
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": True,
    }


async def _read_output(stream: asyncio.StreamReader, max_output_bytes: int) -> str:
    """Drain pipes while retaining a bounded beginning and rolling end."""
    head_limit = (max_output_bytes + 1) // 2
    tail_limit = max_output_bytes // 2
    head = bytearray()
    tail = bytearray()
    total = 0
    while chunk := await stream.read(4096):
        total += len(chunk)
        head_count = min(head_limit - len(head), len(chunk))
        head.extend(chunk[:head_count])
        if tail_limit:
            tail.extend(chunk[head_count:])
            del tail[:-tail_limit]
    if total <= max_output_bytes:
        return (head + tail).decode("utf-8", errors="replace")
    omitted = total - len(head) - len(tail)
    return (
        head.decode("utf-8", errors="replace")
        + f"\n[output truncated: {omitted} bytes omitted]\n"
        + tail.decode("utf-8", errors="replace")
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Stop the shell and its children on timeout or cancellation."""
    if os.name == "nt":
        if process.returncode is None:
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
            except OSError:
                pass
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await process.wait()


async def execute_shell(
    arguments: str, cwd: Path, *, max_output_bytes: int = MAX_OUTPUT_BYTES
) -> str:
    """Validate tool arguments and return a JSON result, including recoverable failures."""
    if max_output_bytes < 1:
        raise ValueError("Shell output limit must be a positive number of bytes.")
    try:
        payload: object = json.loads(arguments)
    except ValueError:
        return json.dumps({"error": "Shell arguments must be a JSON object with a command string."})
    if not isinstance(payload, dict):
        return json.dumps({"error": "Shell arguments must contain only the command field."})
    fields = cast(dict[str, object], payload)
    if set(fields) != {"command"}:
        return json.dumps({"error": "Shell arguments must contain only the command field."})
    command = fields["command"]
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        return json.dumps({"error": "Shell command must be a nonempty string without NUL bytes."})
    invocation = (
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
        if os.name == "nt"
        else ["/bin/sh", "-c", command]
    )
    try:
        process = await asyncio.create_subprocess_exec(
            *invocation,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
    except OSError as error:
        return json.dumps({"error": f"Could not start shell: {error}"})
    assert process.stdout is not None and process.stderr is not None
    readers = [
        asyncio.create_task(_read_output(process.stdout, max_output_bytes)),
        asyncio.create_task(_read_output(process.stderr, max_output_bytes)),
    ]
    try:
        async with asyncio.timeout(COMMAND_TIMEOUT_SECONDS):
            await process.wait()
            stdout, stderr = await asyncio.gather(*readers)
    except (TimeoutError, asyncio.CancelledError) as error:
        await _stop_process(process)
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        if isinstance(error, asyncio.CancelledError):
            raise
        return json.dumps(
            {
                "exit_code": process.returncode,
                "error": f"Command exceeded its {COMMAND_TIMEOUT_SECONDS:g}-second time limit.",
            }
        )
    return json.dumps({"exit_code": process.returncode, "stdout": stdout, "stderr": stderr})
