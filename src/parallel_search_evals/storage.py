"""Filesystem admission, durable replacement and exclusive ownership of run data.

New directories use mode 0700 and files use 0600. Readers open without following
a final symlink and inspect the descriptor before reading, including FIFO cases.
Writers sync file contents before replacing the directory entry and syncing its
parent. A persistent sibling lock inode coordinates canonical aliases of a run;
it must not be unlinked when a writer exits. These guards are shared by result
and configuration storage, independent of model or benchmark behavior.
"""

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


@contextmanager
def directory_handle(path: Path) -> Iterator[int]:
    descriptor = os.open(path, DIRECTORY_FLAGS)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def durable_mkdir(path: Path) -> None:
    if path.is_symlink():
        raise OSError("Output directory must not be a symlink")
    if path.is_dir():
        return
    durable_mkdir(path.parent)
    path.mkdir(mode=0o700, exist_ok=True)
    with directory_handle(path.parent) as descriptor:
        os.fsync(descriptor)


@contextmanager
def regular_file(path: str | Path, flags: int = os.O_RDONLY, *, dir_fd: int | None = None) -> Iterator[BinaryIO]:
    # Nonblocking open reaches fstat even for a FIFO instead of hanging first.
    with os.fdopen(os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=dir_fd), "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError("Run file must be regular")
        yield stream


def atomic_write(path: Path, payload: bytes) -> None:
    durable_mkdir(path.parent)
    temporary = f".{path.name}.{uuid4().hex}.tmp"
    with directory_handle(path.parent) as directory:
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # Replace the entry itself so a destination symlink cannot redirect the write.
            os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
            # Directory sync persists the rename beyond the already-synced file contents.
            os.fsync(directory)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory)


@contextmanager
def run_lock(path: Path) -> Iterator[Path]:
    if path.is_symlink():
        raise OSError("Output directory must not be a symlink")
    path = path.resolve()
    durable_mkdir(path.parent)
    with (
        directory_handle(path.parent) as directory,
        regular_file(f".{path.name}.lock", os.O_RDWR | os.O_CREAT, dir_fd=directory) as lock,
    ):
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another writer owns this run") from None
        if path.exists():
            with directory_handle(path) as owned:
                os.fchmod(owned, 0o700)
        # The canonical sibling inode persists: aliases contend before creating run data.
        yield path
