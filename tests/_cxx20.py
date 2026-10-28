"""Whether Inductor's CPU C++ compiler supports C++20 (torch.compile on CPU needs it), checked once per session.

The compiler is the one Inductor would use: `torch._inductor.cpp_builder.get_cpp_compiler()` (which honours `CXX`),
else `$CXX`, else `c++`.  `pytest_runtest_makereport` (re-exported by conftest) turns the Inductor "#error C++20 or later" failure into a
skip with `SKIP_MESSAGE` when this check is negative; GPU compile cells do not reach the C++ compiler and are unaffected.
"""
import functools
import os
import shlex
import subprocess
import tempfile

import pytest

SKIP_MESSAGE = "torch.compile on CPU needs a C++20 compiler (gcc >= 10); set CXX or load a newer gcc"
#: text of the failure an old compiler produces inside Inductor
FAILURE_MARKERS = ("C++20 or later", "-std=c++20", "std=c++2a")


def _compiler() -> list:
    try:
        from torch._inductor.cpp_builder import get_cpp_compiler
        return shlex.split(get_cpp_compiler())
    except Exception:
        return shlex.split(os.environ.get("CXX") or "c++")


@functools.lru_cache(maxsize=None)
def cpp20_available() -> bool:
    """True when the Inductor CPU compiler compiles a C++20 translation unit (cached for the session)."""
    src = ("#include <version>\n#if __cplusplus < 202002L\n#error C++20 or later\n#endif\n"
           "int main() { return 0; }\n")
    with tempfile.TemporaryDirectory(prefix="cxx20-") as tmp:
        path = os.path.join(tmp, "probe.cpp")
        with open(path, "w") as handle:
            handle.write(src)
        try:
            done = subprocess.run(_compiler() + ["-std=c++20", "-fsyntax-only", path],
                                  capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            return False
    return done.returncode == 0


def is_cpp20_failure(text: str) -> bool:
    """True when `text` is an Inductor CPU compile failure caused by a pre-C++20 compiler."""
    return any(marker in text for marker in FAILURE_MARKERS) and not cpp20_available()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A setup/call failure that is Inductor's "#error C++20 or later" on a host whose CPU compiler lacks C++20 is
    reported as a skip with the install advice."""
    outcome = yield
    report = outcome.get_result()
    if report.failed and call.excinfo is not None and call.when in ("setup", "call"):
        if is_cpp20_failure(str(call.excinfo.value) + str(getattr(report, "longrepr", ""))):
            path, line, _ = item.reportinfo()
            report.outcome = "skipped"
            report.longrepr = (str(path), line or 0, "Skipped: " + SKIP_MESSAGE)
