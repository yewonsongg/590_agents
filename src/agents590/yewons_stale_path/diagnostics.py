"""Small JSON events for diagnosing model rounds without retaining a trace in memory."""

import json
import sys
from hashlib import sha256
from pathlib import Path
from time import monotonic
from typing import cast

from openai.types.responses import Response, ResponseFunctionToolCall

from agents590.yewons_stale_path.tools import ShellStats


class RunDiagnostics:
    """Write events to stderr, keeping credentials and large bodies out of logs."""

    def __init__(self, secrets: tuple[str, ...]) -> None:
        self._secrets = tuple(sorted({value for value in secrets if value}, key=len, reverse=True))
        self._started = monotonic()
        self.requests = 0
        self.responses = 0
        self.tool_calls = 0
        self.input_tokens: int | None = 0
        self.output_tokens: int | None = 0
        self.cached_tokens: int | None = 0

    def text(self, value: str, limit: int = 4096) -> str:
        """Redact before clipping so a truncation boundary cannot expose part of a key."""
        for secret in self._secrets:
            value = value.replace(secret, "[REDACTED]")
        if len(value) <= limit:
            return value
        half = limit // 2
        return value[:half] + "\n[log text truncated]\n" + value[-half:]

    def emit(self, event: str, **fields: object) -> None:
        record = {
            "event": event,
            "elapsed_ms": round((monotonic() - self._started) * 1000, 2),
            **fields,
        }
        encoded = json.dumps(record, separators=(",", ":"))
        for secret in self._secrets:
            encoded = encoded.replace(json.dumps(secret)[1:-1], "[REDACTED]")
        print(encoded, file=sys.stderr, flush=True)

    def model_response(self, step: int, response: Response, duration_ms: float) -> None:
        self.responses += 1
        usage = response.usage
        input_tokens = usage.input_tokens if usage is not None else None
        output_tokens = usage.output_tokens if usage is not None else None
        # Gateways can omit optional usage breakdowns even when totals are supplied.
        cached_tokens: int | None = getattr(
            getattr(usage, "input_tokens_details", None), "cached_tokens", None
        )
        cache_write_tokens: int | None = getattr(
            getattr(usage, "input_tokens_details", None), "cache_write_tokens", None
        )
        reasoning_tokens: int | None = getattr(
            getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
        )
        self.input_tokens = self._add(self.input_tokens, input_tokens)
        self.output_tokens = self._add(self.output_tokens, output_tokens)
        self.cached_tokens = self._add(self.cached_tokens, cached_tokens)
        self.emit(
            "model_response",
            step=step,
            response_id=response.id,
            status=response.status,
            duration_ms=round(duration_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            reasoning_tokens=reasoning_tokens,
            tool_calls=sum(item.type == "function_call" for item in response.output),
        )

    def tool_start(self, step: int, call: ResponseFunctionToolCall, cwd: Path) -> None:
        self.tool_calls += 1
        command: object = None
        try:
            arguments: object = json.loads(call.arguments)
            if isinstance(arguments, dict):
                command = cast(dict[str, object], arguments).get("command")
        except ValueError:
            pass
        self.emit(
            "tool_start",
            step=step,
            call_id=call.call_id,
            tool=call.name,
            cwd=str(cwd),
            command=self.text(command) if isinstance(command, str) else None,
            command_sha256=sha256(command.encode()).hexdigest()
            if isinstance(command, str)
            else None,
        )

    def tool_end(
        self,
        step: int,
        call: ResponseFunctionToolCall,
        stats: ShellStats,
        result: str | None,
        error_type: str | None,
        duration_ms: float,
    ) -> None:
        fields = cast(dict[str, object], json.loads(result)) if result is not None else {}
        stderr = fields.get("stderr")
        error = fields.get("error")
        exit_code = fields.get("exit_code")
        status = "completed"
        if error_type is not None:
            status = "cancelled" if error_type == "CancelledError" else "error"
        elif stats.timed_out:
            status = "timeout"
        elif error is not None or exit_code not in (None, 0):
            status = "failed"
        self.emit(
            "tool_end",
            step=step,
            call_id=call.call_id,
            tool=call.name,
            status=status,
            duration_ms=round(duration_ms, 2),
            exit_code=exit_code,
            error=self.text(error, 512) if isinstance(error, str) else None,
            error_type=error_type,
            stdout=stats.stdout.as_dict(),
            stderr=stats.stderr.as_dict(),
            stderr_excerpt=self.text(stderr, 512) if isinstance(stderr, str) else None,
            result_bytes=len(result.encode()) if result is not None else None,
        )

    @staticmethod
    def _add(total: int | None, value: int | None) -> int | None:
        return total + value if total is not None and value is not None else None

    def finish(self, status: str) -> None:
        self.emit(
            "run_end",
            status=status,
            requests=self.requests,
            responses=self.responses,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens if self.responses else None,
            output_tokens=self.output_tokens if self.responses else None,
            cached_tokens=self.cached_tokens if self.responses else None,
        )
