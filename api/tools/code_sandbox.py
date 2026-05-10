from __future__ import annotations
import io
import sys
import traceback
from typing import Any, Dict, Optional
from RestrictedPython import compile_restricted, safe_globals, safe_builtins
from RestrictedPython.PrintCollector import PrintCollector
from tools.base import BaseTool, FailureReason


# Allowed built-ins for sandboxed code
_SAFE_BUILTINS = safe_builtins.copy()
_SAFE_BUILTINS.update({
    "print":    print,
    "range":    range,
    "len":      len,
    "int":      int,
    "float":    float,
    "str":      str,
    "list":     list,
    "dict":     dict,
    "set":      set,
    "tuple":    tuple,
    "sum":      sum,
    "min":      min,
    "max":      max,
    "abs":      abs,
    "round":    round,
    "sorted":   sorted,
    "enumerate":enumerate,
    "zip":      zip,
    "map":      map,
    "filter":   filter,
    "bool":     bool,
    "isinstance": isinstance,
    "type":     type,
})

_BLOCKED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil",
    "importlib", "ctypes", "pickle", "eval", "exec",
    "__import__", "open",
}


class CodeSandboxTool(BaseTool):
    name = "code_sandbox"
    timeout_seconds = 15.0
    max_code_length = 2000  # characters

    def _validate_input(self, code: str = "", **kwargs) -> Optional[str]:
        if not code or not code.strip():
            return "Code cannot be empty"
        if len(code) > self.max_code_length:
            return f"Code too long (max {self.max_code_length} characters)"
        # Check for obviously blocked patterns
        for blocked in _BLOCKED_IMPORTS:
            if f"import {blocked}" in code or f"__import__('{blocked}')" in code:
                return f"Import of '{blocked}' is not allowed in sandbox"
        return None

    def _execute(self, code: str, **kwargs) -> Dict[str, Any]:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        exit_code      = 0
        result_value   = None

        try:
            # Compile with RestrictedPython
            byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")

            if byte_code is None:
                return {
                    "stdout":     "",
                    "stderr":     "Compilation failed: restricted code detected",
                    "exit_code":  1,
                    "result":     None,
                }

            # Set up restricted globals
            globs = safe_globals.copy()
            globs["__builtins__"] = _SAFE_BUILTINS
            globs["_print_"]      = PrintCollector
            globs["_getiter_"]    = iter
            globs["_getattr_"]    = getattr
            globs["_write_"]      = lambda x: x

            # Capture stdout
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            try:
                exec(byte_code, globs)
                # Collect PrintCollector output if used
                if "_print" in globs:
                    collector = globs["_print"]
                    if hasattr(collector, 'txt'):
                        stdout_capture.write(''.join(collector.txt))
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Check if there's a result variable
            result_value = globs.get("result", None)

        except SyntaxError as e:
            exit_code = 1
            stderr_capture.write(f"SyntaxError: {e}")

        except Exception as e:
            exit_code = 1
            stderr_capture.write(traceback.format_exc())

        return {
            "stdout":    stdout_capture.getvalue(),
            "stderr":    stderr_capture.getvalue(),
            "exit_code": exit_code,
            "result":    result_value,
        }