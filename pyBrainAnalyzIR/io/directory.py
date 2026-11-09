"""Recursive import of supported NIRS recording formats."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.io.dot_nirs import loadDotNIRS
from pyBrainAnalyzIR.io.nirx import loadNIRx

__all__ = ["loadDirectory"]


def loadDirectory(
    folder: str | Path,
    folder_hierarchy: Iterable[str] | None = None,
    file_extensions: Iterable[str] | None = None,
    force_load_all: bool = False,
) -> DataSet:
    """Recursively load SNIRF, DotNIRS, and NIRx recordings under ``folder``.

    SNIRF recordings are handled before ``.nirs`` files so a paired NIRx
    export is imported only once. ``folder_hierarchy`` optionally maps the
    final relative directory components to recording metadata keys.
    """
    root = Path(folder)
    if not root.is_dir():
        raise NotADirectoryError(root)
    extensions = (
        {".snirf", ".nirs", ".wl1"}
        if file_extensions is None
        else {_normalise_extension(ext) for ext in file_extensions}
    )
    hierarchy = list(folder_hierarchy or [])
    dataset = DataSet()
    dataset.description = f"Data loaded from {root}"
    loaded_keys = set()
    handled_nirx_dirs = set()

    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda item: (item.suffix.lower() != ".snirf", str(item).casefold()),
    )
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in extensions:
            continue

        if suffix == ".snirf":
            imported = loadNIRx(path)
            _import_recordings(dataset, imported.dataset, root, hierarchy)
            loaded_keys.add(_recording_key(path))
        elif suffix == ".nirs":
            key = _recording_key(path)
            if not force_load_all and key in loaded_keys:
                continue
            if path.with_suffix(".wl1").exists():
                directory_key = path.parent.resolve()
                if directory_key not in handled_nirx_dirs:
                    imported = loadNIRx(path.parent)
                    _import_recordings(dataset, imported.dataset, root, hierarchy)
                    handled_nirx_dirs.add(directory_key)
            else:
                imported = loadDotNIRS(path, force=force_load_all)
                _import_recordings(dataset, imported.dataset, root, hierarchy)
                loaded_keys.add(key)
        elif suffix == ".wl1":
            directory_key = path.parent.resolve()
            if directory_key not in handled_nirx_dirs and not any(
                item.suffix.lower() in {".snirf", ".nirs"}
                for item in path.parent.iterdir()
                if item.is_file()
            ):
                imported = loadNIRx(path.parent)
                _import_recordings(dataset, imported.dataset, root, hierarchy)
                handled_nirx_dirs.add(directory_key)
    return dataset


def _import_recordings(dataset, recordings, root, hierarchy):
    for recording in recordings:
        if hierarchy:
            description = Path(recording.description)
            try:
                relative_parts = description.parent.relative_to(root).parts
            except ValueError:
                relative_parts = ()
            if len(hierarchy) > len(relative_parts):
                raise ValueError(
                    "folder_hierarchy has more entries than relative directory levels"
                )
            for key, value in zip(hierarchy, relative_parts[-len(hierarchy):]):
                recording.meta_data[key] = value
        dataset.import_data(recording)


def _recording_key(path: Path):
    return path.parent.resolve(), path.stem.casefold()


def _normalise_extension(extension: str) -> str:
    value = extension.strip().lower()
    if not value.startswith("."):
        value = f".{value}"
    return value
