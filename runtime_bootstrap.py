"""
Runtime bootstrap helpers for reliable local launches.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


_BOOTSTRAP_SENTINEL = "OCR_RUNTIME_BOOTSTRAPPED"
_PREFERRED_VERSION = "3.11"


def _candidate_interpreters(project_root: Path) -> list[str]:
    candidates = []

    if os.name == "nt":
        for launcher in ("py", "python3.11", "python"):
            resolved = shutil.which(launcher)
            if resolved:
                candidates.append(resolved)
        return list(dict.fromkeys(candidates))

    for name in ("python3.11",):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def _is_python_311(executable: str) -> bool:
    if not os.path.exists(executable) or not os.access(executable, os.X_OK):
        return False

    try:
        result = subprocess.run(
            [executable, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return False

    return result.returncode == 0 and result.stdout.strip() == _PREFERRED_VERSION


def ensure_python_runtime():
    """
    Re-launch the current entry point with Python 3.11 when needed.
    """
    if os.environ.get(_BOOTSTRAP_SENTINEL) == "1":
        return

    if f"{sys.version_info[0]}.{sys.version_info[1]}" == _PREFERRED_VERSION:
        return

    project_root = Path(__file__).resolve().parent
    for executable in _candidate_interpreters(project_root):
        if not _is_python_311(executable):
            continue

        env = os.environ.copy()
        env[_BOOTSTRAP_SENTINEL] = "1"
        os.execve(executable, [executable, sys.argv[0], *sys.argv[1:]], env)

    raise RuntimeError(
        "Python 3.11 is required for this project. "
        "Run it with `python3.11`."
    )
