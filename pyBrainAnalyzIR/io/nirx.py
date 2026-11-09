"""NIRx directory loading, preferring SNIRF and preserving available sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyBrainAnalyzIR.dataclasses.dataset import DataSet
from pyBrainAnalyzIR.io.dot_nirs import _mat_to_python, loadDotNIRS

__all__ = ["loadNIRx"]

_TEXT_SUFFIXES = {".evt", ".hdr", ".inf", ".log", ".txt", ".tri"}
_MAT_SUFFIXES = {".mat"}
_JSON_SUFFIXES = {".json"}


def loadNIRx(path: str | Path) -> DataSet:
    """Load NIRx recordings from a SNIRF file or a recording directory.

    SNIRF is preferred when present. JSON, MATLAB, and text sidecars are
    attached under ``recording.aux_obj["nirx_sidecars"]``. When a matching
    DotNIRS ``.nirs`` file is available, its stimulus and auxiliary streams
    are also used when they are not already represented in the SNIRF.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file() and source.suffix.lower() == ".snirf":
        snirf_files = [source]
        directory = source.parent
    else:
        directory = source if source.is_dir() else source.parent
        snirf_files = sorted(directory.glob("*.snirf"))

    if snirf_files:
        from cedalion.io.snirf import read_snirf

        dataset = DataSet()
        dataset.description = f"NIRx data from {directory}"
        for snirf_file in snirf_files:
            for recording in read_snirf(snirf_file):
                recording.meta_data.setdefault("NIRxSourceFile", str(snirf_file))
                sidecars = _read_sidecars(directory, snirf_file.stem)
                recording.aux_obj["nirx_sidecars"] = {
                    "files": [str(item) for item in _sidecar_paths(directory, snirf_file.stem)],
                    "metadata": sidecars,
                }
                _merge_dotnirs_sidecar(recording, directory / f"{snirf_file.stem}.nirs")
                dataset.import_data(recording)
        return dataset

    if source.is_file() and source.suffix.lower() == ".nirs":
        nirs_files = [source]
    else:
        nirs_files = sorted(directory.glob("*.nirs"))
    if nirs_files:
        dataset = loadDotNIRS(nirs_files, force=True)
        dataset.description = f"NIRx data from {directory}"
        for recording in dataset.dataset:
            nirs_file = Path(recording.description)
            recording.aux_obj["nirx_sidecars"] = {
                "files": [str(item) for item in _sidecar_paths(directory, nirs_file.stem)],
                "metadata": _read_sidecars(directory, nirs_file.stem),
            }
        return dataset

    if list(directory.glob("*.wl1")):
        raise NotImplementedError(
            "This NIRx folder has raw .wl1/.wl2 files but no SNIRF or DotNIRS "
            ".nirs file. Raw wavelength-file import is not supported yet."
        )
    raise FileNotFoundError(
        f"No SNIRF or .nirs recording found in NIRx path: {source}"
    )


def _merge_dotnirs_sidecar(recording, path: Path) -> None:
    if not path.is_file():
        return
    sidecar_dataset = loadDotNIRS(path, force=True)
    if len(sidecar_dataset.dataset) != 1:
        return
    sidecar = sidecar_dataset.dataset[0]
    signal = recording.timeseries.get("amp")
    sidecar_signal = sidecar.timeseries.get("amp")
    same_time = (
        signal is not None
        and sidecar_signal is not None
        and signal.time.size == sidecar_signal.time.size
        and np.allclose(
            signal.time.values,
            sidecar_signal.time.values,
            rtol=0.0,
            atol=1e-6,
        )
    )
    if same_time:
        for name, auxiliary in sidecar.aux_ts.items():
            recording.aux_ts.setdefault(name, auxiliary)
    if recording.stim.empty and not sidecar.stim.empty:
        recording.stim = sidecar.stim.copy()
    for key, value in sidecar.meta_data.items():
        recording.meta_data.setdefault(key, value)
    recording.aux_obj["nirx_sidecars"]["dotnirs_metadata"] = sidecar.aux_obj.get(
        "dotnirs", {}
    )


def _sidecar_paths(directory: Path, stem: str) -> list[Path]:
    common = {"digpts.txt", "stimulus.mat"}
    paths = []
    for item in sorted(directory.iterdir()):
        if not item.is_file() or item.suffix.lower() in {".snirf", ".wl1", ".wl2", ".nirs"}:
            continue
        if item.name.casefold().startswith(stem.casefold()) or item.name.casefold() in common:
            if item.suffix.lower() in _TEXT_SUFFIXES | _MAT_SUFFIXES | _JSON_SUFFIXES:
                paths.append(item)
    return paths


def _read_sidecars(directory: Path, stem: str) -> dict[str, object]:
    metadata = {}
    for path in _sidecar_paths(directory, stem):
        suffix = path.suffix.lower()
        if suffix in _JSON_SUFFIXES:
            with path.open(encoding="utf-8") as file:
                metadata[path.name] = json.load(file)
        elif suffix in _TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            metadata[path.name] = {
                "text": text,
                "key_values": _parse_key_values(text),
            }
        elif suffix in _MAT_SUFFIXES:
            contents = loadmat(path, squeeze_me=True, struct_as_record=False)
            metadata[path.name] = {
                key: _mat_to_python(value)
                for key, value in contents.items()
                if not key.startswith("__")
            }
    return metadata


def _parse_key_values(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip().replace("-", "_").replace(" ", "_")] = value.strip().strip('"')
    return values
