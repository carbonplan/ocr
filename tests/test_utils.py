from unittest.mock import MagicMock

import pytest
from upath import UPath

from ocr.utils import copy_or_upload


@pytest.fixture
def tmpdir_path(tmp_path):
    return tmp_path


def _read_bytes(p: UPath) -> bytes:
    with p.open('rb') as f:
        return f.read()


def _write_bytes(p: UPath, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as f:
        f.write(data)


def test_copy_or_upload_appends_filename_when_dest_is_dir(tmpdir_path):
    src = UPath(tmpdir_path / 'foo.txt')
    _write_bytes(src, b'hello world')

    dest_dir = UPath(tmpdir_path / 'outdir')
    dest_dir.mkdir(parents=True, exist_ok=True)

    copy_or_upload(src, dest_dir)

    dest_file = dest_dir / 'foo.txt'
    assert dest_file.exists()
    assert _read_bytes(dest_file) == b'hello world'


def test_copy_or_upload_overwrite_false_raises_when_dest_exists(tmpdir_path):
    src = UPath(tmpdir_path / 'a.txt')
    _write_bytes(src, b'source')

    dest = UPath(tmpdir_path / 'b.txt')
    _write_bytes(dest, b'existing')

    with pytest.raises(FileExistsError):
        copy_or_upload(src, dest, overwrite=False)

    # Destination file remains unchanged
    assert _read_bytes(dest) == b'existing'


def test_copy_or_upload_uses_server_side_copy_when_available(tmpdir_path, monkeypatch):
    src = UPath(tmpdir_path / 'src.bin')
    _write_bytes(src, b'payload')

    dest = UPath(tmpdir_path / 'dest.bin')

    # Provide a server-side copy implementation and spy on it
    def server_copy(src_path: str, dest_path: str):
        # Mimic FS-level copy
        with open(src_path, 'rb') as r, open(dest_path, 'wb') as w:
            w.write(r.read())

    mock_copy = MagicMock(side_effect=server_copy)

    # Sanity: src/dest share the same fs type
    assert type(src.fs) is type(dest.fs)

    # Inject the copy method only on this instance (hasattr check will pass)
    monkeypatch.setattr(src.fs, 'copy', mock_copy, raising=False)

    copy_or_upload(src, dest)

    # Assert: server-side copy was used and content matches
    mock_copy.assert_called_once_with(str(src), str(dest))
    assert dest.exists()
    assert _read_bytes(dest) == b'payload'


def test_copy_or_upload_streaming_fallback_across_filesystems(tmpdir_path):
    # local -> memory:// (different FS types)
    src = UPath(tmpdir_path / 'file.txt')
    _write_bytes(src, b'data')

    # Use memory filesystem as destination to force streaming path
    dest_dir = UPath('memory://bucket/dir/')
    dest_dir.mkdir(parents=True, exist_ok=True)

    # copy to a directory-like path on a different filesystem
    copy_or_upload(src, dest_dir)

    # appended filename exists on memory filesystem (use same FS instance)
    dest_file = dest_dir / 'file.txt'
    assert dest_file.exists()
    assert _read_bytes(dest_file) == b'data'
