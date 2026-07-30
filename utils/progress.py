"""Consistent terminal progress and console messaging for long-running jobs."""

from __future__ import annotations

import logging
import math
import sys
import time
from contextlib import contextmanager

from tqdm.auto import tqdm


# Dynamic bars stay disabled for library calls by default. Each command-line
# entry point enables them explicitly; concise status lines remain available.
_PROGRESS_ENABLED = False


def configure_progress(enabled: bool = True) -> None:
    global _PROGRESS_ENABLED
    _PROGRESS_ENABLED = bool(enabled)


def progress_enabled() -> bool:
    return _PROGRESS_ENABLED


def track(
    iterable=None,
    *,
    total=None,
    desc: str = "",
    unit: str = "item",
    leave: bool = False,
    **kwargs,
):
    """Return a project-standard tqdm bar or a transparent disabled wrapper."""
    defaults = {
        "dynamic_ncols": True,
        "mininterval": 0.2,
        "maxinterval": 1.0,
        "smoothing": 0.1,
        "ascii": False,
        "disable": not _PROGRESS_ENABLED,
        "leave": leave,
        "unit": unit,
        "desc": desc,
        "file": sys.stderr,
    }
    defaults.update(kwargs)
    defaults["disable"] = bool(defaults.get("disable", False) or not _PROGRESS_ENABLED)
    return tqdm(iterable, total=total, **defaults)


def console(message: str, *, level: str = "INFO") -> None:
    """Write a timestamped line without corrupting active progress bars."""
    line = f"{time.strftime('%H:%M:%S')} | {level.upper():7s} | {message}"
    if _PROGRESS_ENABLED:
        tqdm.write(line, file=sys.stderr)
    else:
        print(line, file=sys.stderr, flush=True)


def metric_line(metrics: dict, names=("auc", "pr_auc", "f1", "kappa")) -> str:
    parts = []
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        label = "PR-AUC" if name == "pr_auc" else name.upper()
        parts.append(f"{label}={numeric:.4f}")
    return " | ".join(parts)


def window_count(height: int, width: int, size: int) -> int:
    return int(math.ceil(height / size) * math.ceil(width / size))


@contextmanager
def timed_task(description: str, *, level: str = "INFO"):
    """Announce an opaque task and report its elapsed wall time."""
    started = time.monotonic()
    announce = _PROGRESS_ENABLED
    if announce:
        console(f"开始：{description}", level=level)
    try:
        yield
    except Exception:
        if announce:
            console(
                f"失败：{description}（耗时 {time.monotonic() - started:.1f}s）",
                level="ERROR",
            )
        raise
    else:
        if announce:
            console(f"完成：{description}（耗时 {time.monotonic() - started:.1f}s）")


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that coexists cleanly with nested tqdm bars."""

    def emit(self, record):
        try:
            message = self.format(record)
            if _PROGRESS_ENABLED:
                tqdm.write(message, file=sys.stderr)
            else:
                print(message, file=sys.stderr, flush=True)
        except Exception:
            self.handleError(record)
