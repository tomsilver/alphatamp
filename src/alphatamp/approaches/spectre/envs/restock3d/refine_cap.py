"""A real per-candidate wall-clock cap for Restock3D refinement.

The substrate's ``refinement_timeout_s`` is cooperative — checked only at
``BacktrackingRefiner`` step-recursion entry, never mid-step or between a step's
sampling attempts — so an over-budget candidate can overshoot by ~one step's motion-
planning attempts. ``hard_timeout`` installs a real SIGALRM so an over-budget
``refiner(...)`` is actually interrupted.

Limitation: SIGALRM unwinds only at Python-bytecode boundaries, so a long in-flight
PyBullet C call finishes before the exception is raised (the pure-Python
controller/refiner loop is interrupted promptly). SIGALRM is main-thread only; it works
inside a spawn worker process (whose main thread runs the refine) but NOT inside a
background thread.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from typing import Iterator


class RefineTimeout(Exception):
    """Raised when a refinement exceeds its hard wall-clock cap."""


@contextmanager
def hard_timeout(seconds: float) -> Iterator[None]:
    """Interrupt the wrapped block with :class:`RefineTimeout` after ``seconds``
    (SIGALRM).

    ``seconds <= 0`` disables the cap (yields without arming the timer).
    """
    if seconds <= 0:
        yield
        return

    def _handler(signum: int, frame: object) -> None:
        raise RefineTimeout(f"refinement exceeded hard cap of {seconds:.1f}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)
