"""
utils/version.py — Build/version stamp for the running process.

Answers the recurring question "is the live process running my latest edits?"
without needing git on PATH. Strategy, in order of preference:

  1. git short hash + dirty flag, if a .git dir and git binary are available
  2. otherwise, a content fingerprint: the most-recent mtime across all
     tracked .py files plus a short hash of their combined size/mtime, so any
     code edit changes the stamp deterministically

Call build_stamp() at startup and log it. Compare the logged value before and
after a restart to confirm new code was loaded.
"""
from __future__ import annotations
import hashlib
import os
import subprocess
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SKIP_DIRS = {".git", ".venv", ".venv-win", "__pycache__", "logs",
              "backtesting", "Kindle_files", "node_modules"}


def _git_stamp() -> str | None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5,
        )
        if head.returncode != 0:
            return None
        short = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5,
        )
        dirty = "+dirty" if status.stdout.strip() else ""
        return f"git:{short}{dirty}"
    except Exception:
        return None


def _content_stamp() -> str:
    """Fingerprint all .py files by (relpath, size, mtime). Deterministic."""
    latest_mtime = 0.0
    h = hashlib.sha1()
    for dp, dns, fns in os.walk(_ROOT):
        dns[:] = [d for d in dns if d not in _SKIP_DIRS]
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dp, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            rel = os.path.relpath(fp, _ROOT)
            h.update(f"{rel}:{st.st_size}:{st.st_mtime_ns}".encode())
            latest_mtime = max(latest_mtime, st.st_mtime)
    digest = h.hexdigest()[:8]
    newest = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S") if latest_mtime else "unknown"
    return f"content:{digest}  newest_py={newest}"


def build_stamp() -> str:
    """Return a human-readable build identifier for the current source tree."""
    return _git_stamp() or _content_stamp()


if __name__ == "__main__":
    print(build_stamp())
