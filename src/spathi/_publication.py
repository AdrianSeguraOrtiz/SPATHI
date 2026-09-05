"""Atomic, never-overwrite publication primitives shared by SPATHI workflows."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004

# ``renameat2`` entered Linux through different architecture-specific syscall
# tables. These values come from the corresponding Linux UAPI ``unistd``
# headers. They are used only when libc predates the glibc 2.28 wrapper; modern
# glibc and musl installations take the named-function path instead.
_LINUX_RENAMEAT2_SYSCALLS = {
    "aarch64": 276,
    "amd64": 316,
    "arm": 382,
    "arm64": 276,
    "armv6l": 382,
    "armv7l": 382,
    "armv8l": 382,
    "i386": 353,
    "i486": 353,
    "i586": 353,
    "i686": 353,
    "loongarch64": 276,
    "ppc64": 357,
    "ppc64le": 357,
    "riscv64": 276,
    "s390x": 347,
    "x86": 353,
    "x86_64": 316,
}


def path_is_occupied(path: Path) -> bool:
    """Treat broken symbolic links as occupied never-overwrite paths."""

    return path.exists() or path.is_symlink()


def _process_c_library() -> Any:
    """Load process C symbols while preserving the called function's errno."""

    return ctypes.CDLL(None, use_errno=True)


def _linux_machine() -> str:
    """Return the normalized Linux architecture used for syscall selection."""

    return os.uname().machine.lower()


def _linux_rename_no_replace(source: Path, destination: Path) -> int:
    """Call Linux ``renameat2(RENAME_NOREPLACE)`` through libc or ``syscall``."""

    library = _process_c_library()
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    ctypes.set_errno(0)
    try:
        rename_no_replace = library.renameat2
    except AttributeError:
        machine = _linux_machine()
        syscall_number = _LINUX_RENAMEAT2_SYSCALLS.get(machine)
        if syscall_number is None:
            raise RuntimeError(
                "atomic no-replace directory publication requires either libc renameat2 "
                f"or a known Linux syscall number; unsupported architecture: {machine!r}"
            ) from None
        try:
            syscall = library.syscall
        except AttributeError as exc:
            raise RuntimeError(
                "atomic no-replace directory publication requires renameat2 or libc syscall"
            ) from exc
        # ``syscall`` is variadic, so every argument is explicitly wrapped and
        # ``argtypes`` must remain unset.
        syscall.restype = ctypes.c_long
        return int(
            syscall(
                ctypes.c_long(syscall_number),
                ctypes.c_long(_AT_FDCWD),
                ctypes.c_char_p(encoded_source),
                ctypes.c_long(_AT_FDCWD),
                ctypes.c_char_p(encoded_destination),
                ctypes.c_ulong(_RENAME_NOREPLACE),
            )
        )

    rename_no_replace.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    rename_no_replace.restype = ctypes.c_int
    return int(
        rename_no_replace(
            _AT_FDCWD,
            encoded_source,
            _AT_FDCWD,
            encoded_destination,
            _RENAME_NOREPLACE,
        )
    )


def publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename ``source`` while refusing every occupied destination.

    Plain POSIX ``rename`` may replace an empty destination directory, leaving a
    race between an existence check and publication. Linux and macOS expose
    no-replace rename flags; Windows already rejects every existing destination.
    Refuse publication on an unsupported platform instead of weakening SPATHI's
    never-overwrite contract.
    """

    if sys.platform.startswith("linux"):
        result = _linux_rename_no_replace(source, destination)
    elif sys.platform == "darwin":  # pragma: no cover - exercised by platform CI
        library = _process_c_library()
        ctypes.set_errno(0)
        try:
            rename_no_replace = library.renamex_np
        except AttributeError as exc:
            raise RuntimeError(
                "atomic no-replace directory publication requires renamex_np on macOS"
            ) from exc
        rename_no_replace.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename_no_replace.restype = ctypes.c_int
        result = rename_no_replace(
            os.fsencode(source),
            os.fsencode(destination),
            _RENAME_EXCL,
        )
    elif os.name == "nt":  # pragma: no cover - exercised by platform CI
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                exc.errno or errno.EEXIST,
                "output path already exists and will not be overwritten",
                destination,
            ) from None
        return
    else:  # pragma: no cover - platform-specific safety guard
        raise RuntimeError(
            f"atomic no-replace directory publication is unsupported on {sys.platform!r}"
        )

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "output path already exists and will not be overwritten",
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def preflight_atomic_publication(parent: Path) -> None:
    """Verify no-replace and successful publication on the target filesystem.

    The probe runs before inputs are loaded, so an unsupported operating system,
    libc, kernel, or filesystem cannot waste a complete scientific workflow.
    """

    with TemporaryDirectory(prefix=".spathi-publication-probe-", dir=parent) as probe_name:
        probe = Path(probe_name)
        source = probe / "source"
        destination = probe / "destination"
        source.mkdir()
        destination.mkdir()

        try:
            publish_directory_no_replace(source, destination)
        except FileExistsError:
            pass
        else:  # pragma: no cover - defensive guard around platform primitives
            raise RuntimeError("atomic publication preflight replaced an occupied destination")
        if not source.is_dir() or not destination.is_dir():
            raise RuntimeError("atomic publication preflight changed paths after a rejected rename")

        destination.rmdir()
        publish_directory_no_replace(source, destination)
        if path_is_occupied(source) or not destination.is_dir():
            raise RuntimeError("atomic publication preflight did not publish the source directory")


__all__ = [
    "path_is_occupied",
    "preflight_atomic_publication",
    "publish_directory_no_replace",
]
