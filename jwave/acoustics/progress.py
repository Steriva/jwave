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

import sys
import threading
from typing import Any

_PROGRESS_LOCK = threading.Lock()
_PROGRESS_STATE: dict[str, Any] = {"bar": None, "last": -1, "total": None}

# Cap host callbacks to avoid overhead and Jupyter/ZMQ issues.
_MAX_PROGRESS_UPDATES = 100


def progress_stride(total_steps: int) -> int:
    """Return a stride that limits progress updates to ~100 per simulation."""
    return max(1, int(total_steps) // _MAX_PROGRESS_UPDATES)


def report_simulation_progress(step: int, total: int, description: str) -> None:
    """Host callback used by ``jax.debug.callback`` during time stepping."""
    step = int(step)
    total = int(total)

    with _PROGRESS_LOCK:
        try:
            from tqdm import tqdm
        except ImportError:
            if step == 0 or step + 1 == total or step % max(1, total // 20) == 0:
                print(
                    f"{description}: step {step + 1}/{total}",
                    file=sys.stderr,
                    flush=True,
                )
            return

        bar = _PROGRESS_STATE["bar"]
        if bar is None or _PROGRESS_STATE["total"] != total:
            if bar is not None:
                bar.close()
            # Use stderr tqdm (not tqdm.auto) to avoid ipywidgets/ZMQ in notebooks.
            bar = tqdm(
                total=total,
                desc=description,
                unit="step",
                file=sys.stderr,
                dynamic_ncols=True,
                mininterval=0.25,
            )
            _PROGRESS_STATE["bar"] = bar
            _PROGRESS_STATE["last"] = -1
            _PROGRESS_STATE["total"] = total

        last = _PROGRESS_STATE["last"]
        if step > last:
            bar.update(step - last)
            _PROGRESS_STATE["last"] = step

        if step + 1 >= total:
            remaining = total - bar.n
            if remaining > 0:
                bar.update(remaining)
            bar.close()
            _PROGRESS_STATE["bar"] = None
            _PROGRESS_STATE["last"] = -1
            _PROGRESS_STATE["total"] = None


def reset_simulation_progress() -> None:
    """Close any open progress bar, e.g. after an interrupted run."""
    with _PROGRESS_LOCK:
        bar = _PROGRESS_STATE.get("bar")
        if bar is not None:
            bar.close()
        _PROGRESS_STATE["bar"] = None
        _PROGRESS_STATE["last"] = -1
        _PROGRESS_STATE["total"] = None
