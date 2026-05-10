from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from ulid import ULID


class FailureReason(str, Enum):
    TIMEOUT       = "timeout"
    EMPTY_RESULTS = "empty_results"
    MALFORMED     = "malformed_input"
    EXECUTION     = "execution_error"
    RETRY_LIMIT   = "retry_limit_exceeded"


@dataclass
class ToolResult:
    call_id:        str
    tool_name:      str
    success:        bool
    data:           Any                        # actual result on success
    failure_reason: Optional[FailureReason]   # set on failure
    error_message:  Optional[str]
    latency_ms:     float
    retry_number:   int = 0

    def is_empty(self) -> bool:
        if self.data is None:
            return True
        if isinstance(self.data, (list, dict, str)) and len(self.data) == 0:
            return True
        return False


class BaseTool(ABC):
    name: str = "base_tool"
    timeout_seconds: float = 10.0

    def run(self, retry_number: int = 0, **kwargs) -> ToolResult:
        """
        Entry point. Handles timing, catches all exceptions,
        enforces the failure contract.
        """
        call_id = str(ULID())
        start = time.time()

        # Validate input first
        validation_error = self._validate_input(**kwargs)
        if validation_error:
            latency = round((time.time() - start) * 1000, 2)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                data=None,
                failure_reason=FailureReason.MALFORMED,
                error_message=validation_error,
                latency_ms=latency,
                retry_number=retry_number,
            )

        try:
            data = self._execute(**kwargs)
            latency = round((time.time() - start) * 1000, 2)

            if data is None or (isinstance(data, (list, dict, str)) and len(data) == 0):
                return ToolResult(
                    call_id=call_id,
                    tool_name=self.name,
                    success=False,
                    data=data,
                    failure_reason=FailureReason.EMPTY_RESULTS,
                    error_message="Tool returned empty results",
                    latency_ms=latency,
                    retry_number=retry_number,
                )

            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=True,
                data=data,
                failure_reason=None,
                error_message=None,
                latency_ms=latency,
                retry_number=retry_number,
            )

        except TimeoutError:
            latency = round((time.time() - start) * 1000, 2)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                data=None,
                failure_reason=FailureReason.TIMEOUT,
                error_message=f"Tool timed out after {self.timeout_seconds}s",
                latency_ms=latency,
                retry_number=retry_number,
            )

        except Exception as e:
            latency = round((time.time() - start) * 1000, 2)
            return ToolResult(
                call_id=call_id,
                tool_name=self.name,
                success=False,
                data=None,
                failure_reason=FailureReason.EXECUTION,
                error_message=str(e),
                latency_ms=latency,
                retry_number=retry_number,
            )

    @abstractmethod
    def _execute(self, **kwargs) -> Any:
        """Implement actual tool logic here."""
        pass

    def _validate_input(self, **kwargs) -> Optional[str]:
        """Return an error string if input is invalid, else None."""
        return None