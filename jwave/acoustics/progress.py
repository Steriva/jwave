# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import os
import threading
import time

_PROGRESS_LOCK = threading.Lock()
_LAST_REPORTED_MILESTONE = -1.0
_START_TIME: float | None = None


def progress_stride(total_steps: int, interval_pct: float = 5.0) -> int:
    """Return a scan stride that triggers progress checks every ``interval_pct``."""
    total_steps = int(total_steps)
    interval_pct = max(0.1, min(100.0, float(interval_pct)))
    target_callbacks = max(1, int(round(100.0 / interval_pct)))
    return max(1, total_steps // target_callbacks)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    seconds = max(0, int(seconds + 0.5))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def report_simulation_progress(
    step: int,
    total: int,
    description: str,
    interval_pct: float = 5.0,
) -> None:
    """Host callback used by ``jax.debug.callback`` during time stepping.

    Always writes directly to file descriptor 2. JAX executes this on XLA worker
    threads where ``sys.stderr`` and tqdm are unsafe inside Jupyter kernels.
    """
    step = int(step)
    total = int(total)
    interval_pct = max(0.1, min(100.0, float(interval_pct)))

    with _PROGRESS_LOCK:
        global _LAST_REPORTED_MILESTONE, _START_TIME

        percent = 100.0 * (step + 1) / total
        is_first = step == 0
        is_last = step + 1 >= total
        milestone = 100.0 if is_last else (int(percent // interval_pct) * interval_pct)

        if not is_first and not is_last and milestone <= _LAST_REPORTED_MILESTONE:
            return

        eta_suffix = ""
        if _START_TIME is not None and not is_last:
            elapsed = time.perf_counter() - _START_TIME
            steps_done = step + 1
            remaining = elapsed / steps_done * (total - steps_done)
            eta_suffix = f" | ETA {_format_duration(remaining)}"

        line = (
            f"{description}: step {step + 1}/{total} "
            f"({percent:.0f}%){eta_suffix}"
        )
        if is_last:
            message = f"\r{line}\n"
            _LAST_REPORTED_MILESTONE = -1.0
            _START_TIME = None
        else:
            # Pad so a shorter update fully overwrites the previous line.
            message = f"\r{line:<96}"
            _LAST_REPORTED_MILESTONE = milestone
        os.write(2, message.encode())


def reset_simulation_progress() -> None:
    """Reset progress state before a new simulation."""
    global _LAST_REPORTED_MILESTONE, _START_TIME
    with _PROGRESS_LOCK:
        _LAST_REPORTED_MILESTONE = -1.0
        _START_TIME = time.perf_counter()
