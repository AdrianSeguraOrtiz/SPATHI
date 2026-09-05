from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import pytest

import spathi._publication as publication_module


def test_atomic_publication_refuses_an_existing_empty_directory(tmp_path: Path) -> None:
    source = tmp_path / "staged"
    destination = tmp_path / "published"
    source.mkdir()
    destination.mkdir()
    marker = source / "result.txt"
    marker.write_text("complete", encoding="utf-8")

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        publication_module.publish_directory_no_replace(source, destination)

    assert marker.read_text(encoding="utf-8") == "complete"
    assert list(destination.iterdir()) == []


def test_atomic_publication_preflight_uses_the_target_filesystem(tmp_path: Path) -> None:
    publication_parent = tmp_path / "publication-parent"
    publication_parent.mkdir()

    publication_module.preflight_atomic_publication(publication_parent)

    assert list(publication_parent.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux syscall fallback")
def test_linux_atomic_publication_falls_back_when_libc_has_no_renameat2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_library = ctypes.CDLL(None, use_errno=True)

    class SyscallOnlyLibrary:
        syscall = real_library.syscall

    monkeypatch.setattr(publication_module, "_process_c_library", SyscallOnlyLibrary)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        publication_module.publish_directory_no_replace(source, destination)
    assert source.is_dir()
    assert destination.is_dir()

    destination.rmdir()
    publication_module.publish_directory_no_replace(source, destination)
    assert not source.exists()
    assert destination.is_dir()


def test_linux_syscall_fallback_rejects_an_unknown_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyscallOnlyLibrary:
        def syscall(self) -> int:
            raise AssertionError("unknown architectures must fail before invoking syscall")

    monkeypatch.setattr(publication_module, "_process_c_library", SyscallOnlyLibrary)
    monkeypatch.setattr(publication_module, "_linux_machine", lambda: "unknown-cpu")

    with pytest.raises(RuntimeError, match="unsupported architecture"):
        publication_module._linux_rename_no_replace(Path("source"), Path("destination"))
