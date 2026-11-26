"""Small runtime helpers for safe, reproducible CLI execution."""
from __future__ import annotations

import fcntl
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class RunAlreadyActiveError(RuntimeError):
    """Raised when another run is using the same report directory."""


@contextmanager
def exclusive_run_lock(path: Path) -> Iterator[None]:
    """Hold a non-blocking process lock for a report directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunAlreadyActiveError(
                f"another experiment is already running (lock: {path})"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_run_metadata(path: Path, metadata: dict[str, object]) -> None:
    """Atomically write metadata for a completed run."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    temporary_path.replace(path)
