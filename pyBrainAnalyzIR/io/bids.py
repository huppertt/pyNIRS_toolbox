from pathlib import Path
import copy
import re
import shutil
import subprocess
import pandas as pd
import numpy as np
import json
from collections import OrderedDict
from cedalion.dataclasses.geometry import PointType
import cedalion.xrutils as xrutils

from cedalion.io.snirf import read_snirf, write_snirf
from datetime import datetime
from typing import Any

from cedalion import units
import pyBrainAnalyzIR
import pyBrainAnalyzIR.dataclasses.dataset as dataset


"""# BIDS Dataset
This dataset is organized according to the Brain Imaging Data Structure (BIDS) standard. For more information about BIDS, please visit [https://bids.neuroimaging.io/](https://bids.neuroimaging.io/).  # noqa: E501
"""


def validate_bids_folder(folder: str | Path, timeout: float = 300) -> dict[str, Any]:
    """Validate a BIDS dataset using the official BIDS Validator CLI.

    The validator runs locally and emits its JSON report. Install the official
    ``bids-validator`` or ``bids-validator-deno`` command, or install Deno to
    run the validator from JSR. The first Deno run may need network access to
    download the validator.

    Args:
        folder: Path to the BIDS dataset root.
        timeout: Maximum time in seconds to wait for validation.

    Returns:
        The JSON report produced by the BIDS Validator.

    Raises:
        FileNotFoundError: If the dataset folder or validator executable is
            unavailable.
        ValueError: If ``folder`` is not a directory or ``timeout`` is not
            positive.
        RuntimeError: If the validator does not produce a JSON report.
        subprocess.TimeoutExpired: If validation exceeds ``timeout``.
    """
    dataset_path = Path(folder).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"BIDS folder does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise ValueError(f"BIDS path must be a folder: {dataset_path}")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")

    dataset_path = dataset_path.resolve()
    validator = shutil.which("bids-validator") or shutil.which("bids-validator-deno")
    if validator:
        command = [validator, str(dataset_path), "--format", "json"]
    else:
        deno = shutil.which("deno")
        if not deno:
            raise FileNotFoundError(
                "The BIDS Validator was not found. Install the official "
                "'bids-validator' command or the Deno runtime."
            )
        command = [
            deno, "run", "-ERWN", "jsr:@bids/validator",
            str(dataset_path), "--format", "json",
        ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        details = result.stderr.strip() or result.stdout.strip() or "No output"
        raise RuntimeError(
            "The BIDS Validator did not return a JSON report "
            f"(exit status {result.returncode}): {details}"
        ) from exc

    if not isinstance(report, dict):
        raise RuntimeError("The BIDS Validator JSON report must be an object")
    return report


def write_bids_data(data, path: str | Path,
                    participant_id: str,
                    run: str,
                    task: str = 'func',
                    session: str | None = None,
                    overwrite_snirf: bool = False,
                    include_derivatives: bool = True,
                    stimulus_presentation: dict[str, Any] | None = None,
                    event_descriptions: dict[str, str] | None = None):
    """Write individual data and NIRS channel/optode sidecars in BIDS format."""
    participant_id = _bids_label(participant_id)
    task = _bids_label(task)
    run = _bids_label(run)
    if session is not None:
        session = _bids_label(session)

    folder, filename, snirffile = _bids_nirs_paths(
        path, participant_id, run, task, session
    )
    legacy_snirffile = folder / (filename + "_fnirs.snirf")
    _migrate_legacy_bids_file(legacy_snirffile, snirffile)

    # TODO - the SNIRF writer assumes this order. It shouldn't, but it isn't my code,
    # so I just need to conform to it.
    stim = data.stim[["trial_type", "onset", "duration", "value"]]

    if snirffile.exists():
        _assert_snirf_raw_data_matches(snirffile, data)
    else:
        snirffile.parent.mkdir(parents=True, exist_ok=True)
        _write_snirf_with_timeseries(snirffile, data, ["amp"], stim)
    _write_bids_nirs_sidecars(
        data,
        folder,
        filename,
        participant_id,
        session,
    )
    events_path = folder / (filename + "_events.tsv")
    write_bids_stim(
        data.stim.copy(),
        path=events_path,
        stimulus_presentation=stimulus_presentation,
        event_descriptions=event_descriptions,
    )
    (folder / (filename + "_stim.tsv")).unlink(missing_ok=True)
    (folder / (filename + "_stim.json")).unlink(missing_ok=True)

    derivative_keys = _derivative_timeseries_keys(data)
    if include_derivatives and derivative_keys:
        derivative_file = _derivative_snirf_path(Path(path), snirffile)
        derivative_file.parent.mkdir(parents=True, exist_ok=True)
        if overwrite_snirf or not derivative_file.exists():
            _write_snirf_with_timeseries(derivative_file, data, derivative_keys, stim)

    info = {
        'TaskName': task,
        'Description': 'Raw fNIRS amplitude time series for this recording.',
        'Version': data.meta_data.get('Version', 'unknown'),
        'MeasurementDate': data.meta_data.get('MeasurementDate', 'unknown'),
        'MeasurementTime': data.meta_data.get('MeasurementTime', 'unknown'),
        'SNIRF_createDate': data.meta_data.get(
            'SNIRF_CreateDate', datetime.now().strftime('%Y-%m-%d')
        ),
        'SNIRF_createTime': data.meta_data.get(
            'SNIRF_createTime', f"{datetime.now().strftime('%H:%M:%S')}Z"
        ),
        'filedescription': data.meta_data.get('filedescription', 'unknown'),
    }
    snirffilejson = folder / (filename + "_nirs.json")
    with open(snirffilejson, 'w') as f:
        json.dump(info, f, indent=4)
    (folder / (filename + "_fnirs.json")).unlink(missing_ok=True)


def _bids_nirs_paths(
    path: str | Path,
    participant_id: str,
    run: str,
    task: str,
    session: str | None,
) -> tuple[Path, str, Path]:
    participant_id = _bids_label(participant_id)
    task = _bids_label(task)
    run = _bids_label(run)
    if session is not None:
        session = _bids_label(session)

    if session is None:
        folder = Path(path) / f"sub-{participant_id}"
        filename = f"sub-{participant_id}_task-{task}_run-{run}"
    else:
        folder = Path(path) / f"sub-{participant_id}" / f"ses-{session}"
        filename = f"sub-{participant_id}_ses-{session}_task-{task}_run-{run}"
    return folder, filename, folder / f"{filename}_nirs.snirf"


def _arrays_equal(left, right) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def _assert_snirf_raw_data_matches(snirf_file: Path, data) -> None:
    """Reject reuse of a raw SNIRF path for a different amplitude time course."""
    recordings = read_snirf(snirf_file)
    if not recordings:
        raise ValueError(f"Cannot verify existing SNIRF file: {snirf_file}")
    if "amp" not in getattr(recordings[0], "timeseries", {}):
        raise ValueError(
            f"Existing SNIRF file has no raw 'amp' time course: {snirf_file}"
        )

    existing = recordings[0]["amp"]
    incoming = data["amp"]
    existing_dims = getattr(existing, "dims", None)
    incoming_dims = getattr(incoming, "dims", None)
    if existing_dims is not None and incoming_dims is not None:
        if set(existing_dims) != set(incoming_dims):
            raise ValueError(
                f"Raw amplitude dimensions differ from existing SNIRF file: {snirf_file}"
            )
        if any(
            existing.sizes[dim] != incoming.sizes[dim]
            for dim in incoming_dims
        ):
            raise ValueError(
                f"Raw amplitude dimensions differ from existing SNIRF file: {snirf_file}"
            )
        existing = existing.transpose(*incoming_dims)
    elif existing_dims != incoming_dims:
        raise ValueError(
            f"Raw amplitude dimensions differ from existing SNIRF file: {snirf_file}"
        )

    if hasattr(existing, "pint") and hasattr(incoming, "pint"):
        existing = existing.pint.to(incoming.pint.units)
        existing_values = np.asarray(existing.pint.magnitude)
        incoming_values = np.asarray(incoming.pint.magnitude)
    else:
        existing_values = np.asarray(getattr(existing, "values", existing))
        incoming_values = np.asarray(getattr(incoming, "values", incoming))

    if existing_values.shape != incoming_values.shape or not _arrays_equal(
        existing_values, incoming_values
    ):
        raise ValueError(
            "Raw amplitude time course differs from the existing SNIRF file; "
            f"raw SNIRF files are immutable: {snirf_file}"
        )

    existing_coords = getattr(existing, "coords", {})
    incoming_coords = getattr(incoming, "coords", {})
    if ("time" in existing_coords) != ("time" in incoming_coords):
        raise ValueError(
            f"Raw amplitude time coordinates differ from existing SNIRF file: {snirf_file}"
        )
    if "time" in existing_coords and not _arrays_equal(
        np.asarray(existing_coords["time"].values),
        np.asarray(incoming_coords["time"].values),
    ):
        raise ValueError(
            f"Raw amplitude time coordinates differ from existing SNIRF file: {snirf_file}"
        )


def _write_bids_nirs_sidecars(
    data,
    folder: Path,
    filename: str,
    participant_id: str,
    session: str | None,
) -> None:
    """Write the linked channels, optodes, and coordinate-system files."""
    ts = data["amp"]
    required_coords = {"source", "detector", "wavelength"}
    missing_coords = required_coords.difference(ts.coords)
    missing_dims = {"channel", "wavelength"}.difference(ts.dims)
    if missing_coords or missing_dims:
        raise ValueError(
            "Cannot write BIDS NIRS channel sidecar; missing timeseries "
            f"coordinate(s): {sorted(missing_coords)}, dimension(s): "
            f"{sorted(missing_dims)}"
        )

    sources = [str(value) for value in ts.source.values]
    detectors = [str(value) for value in ts.detector.values]
    wavelengths = ts.wavelength.values
    if (
        len(sources) != ts.sizes["channel"]
        or len(detectors) != ts.sizes["channel"]
        or len(wavelengths) == 0
    ):
        raise ValueError(
            "Cannot write BIDS NIRS channel sidecar; source, detector, and "
            "wavelength coordinates do not match the timeseries dimensions"
        )

    channel_units = format(ts.pint.units, "~")
    source_indices = {
        label: index for index, label in enumerate(dict.fromkeys(sources), start=1)
    }
    detector_indices = {
        label: index for index, label in enumerate(dict.fromkeys(detectors), start=1)
    }
    channel_rows = [
        {
            "name": str(ts.channel.values[channel_index]),
            "type": "NIRSCWAMPLITUDE",
            "source": source,
            "detector": detector,
            "wavelength_nominal": wavelength.item()
            if hasattr(wavelength, "item") else wavelength,
            "units": channel_units,
            "description": (
                "fNIRS Channel between Optodes Source index "
                f"{source_indices[source]} to Detector Index "
                f"{detector_indices[detector]} at wavelength "
                f"{wavelength:g}nm"
            ),
        }
        for channel_index, (source, detector) in enumerate(zip(sources, detectors))
        for wavelength in wavelengths
    ]
    channels = pd.DataFrame(
        channel_rows,
        columns=[
            "name",
            "type",
            "source",
            "detector",
            "wavelength_nominal",
            "units",
            "description",
        ],
    )

    optode_types: dict[str, str] = {}
    for label in sources:
        optode_types[label] = "source"
    for label in detectors:
        if optode_types.get(label) == "source":
            raise ValueError(f"NIRS optode label is both a source and detector: {label}")
        optode_types[label] = "detector"

    optode_labels = list(optode_types)
    positions = (
        data.geo3d.sel(label=optode_labels)
        .pint.to(units.millimeter)
        .pint.magnitude
    )
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError(
            "Cannot write BIDS optode sidecar; expected 2D or 3D optode positions"
        )

    optode_rows = []
    for index, label in enumerate(optode_labels):
        position = positions[index]
        optode_type = optode_types[label]
        optode_type_labels = sources if optode_type == "source" else detectors
        type_index = list(dict.fromkeys(optode_type_labels)).index(label) + 1
        optode_rows.append({
            "name": label,
            "type": optode_type,
            "x": position[0],
            "y": position[1],
            "z": position[2] if positions.shape[1] == 3 else "n/a",
            "description": (
                f"fNIRS {optode_type.capitalize()} Optode index {type_index}"
            ),
        })

    channels.to_csv(folder / f"{filename}_channels.tsv", sep="\t", index=False)
    sidecar_prefix = f"sub-{participant_id}"
    if session is not None:
        sidecar_prefix += f"_ses-{session}"
    pd.DataFrame(
        optode_rows,
        columns=["name", "type", "x", "y", "z", "description"],
    ).to_csv(folder / f"{sidecar_prefix}_optodes.tsv", sep="\t", index=False)

    with open(folder / f"{filename}_channels.json", "w") as f:
        json.dump({
            "Description": (
                "fNIRS channels defined by source-detector pairs and nominal "
                "wavelengths."
            )
        }, f, indent=4)
    with open(folder / f"{sidecar_prefix}_optodes.json", "w") as f:
        json.dump({
            "Description": (
                "fNIRS source and detector optode positions in the coordinate "
                "system described by the accompanying coordsystem.json."
            )
        }, f, indent=4)

    coordsystem = {
        "NIRSCoordinateSystem": "Other",
        "NIRSCoordinateUnits": "mm",
        "NIRSCoordinateSystemDescription": (
            "Coordinates are expressed in the recording's native optode "
            "coordinate frame. A standardized reference system is not inferred."
        ),
    }
    with open(folder / f"{sidecar_prefix}_coordsystem.json", "w") as f:
        json.dump(coordsystem, f, indent=4)


def write_bids_dataset(dataset, path: str | Path,
                       readme: dict | None = None,
                       dataset_description: dict | None = None,
                       overwrite_snirf: bool = False,
                       include_derivatives: bool = True):

    """Write the dataset to a BIDS-compliant directory.

    Existing raw SNIRF files are immutable. Their amplitude time courses must
    match the recordings being saved; metadata and BIDS sidecars are refreshed
    without rewriting the raw files. ``overwrite_snirf`` only controls existing
    derivative SNIRF files.
    """

    if readme is None:
        readme = dataset.readme()
    else:
        readme = {**dataset.readme(), **readme}
    readme_include = dataset.readme_include()
    readme = {
        section: content
        for section, content in readme.items()
        if readme_include.get(section, True)
    }
    if dataset_description is None:
        dataset_description = dataset.data_description()
    else:
        dataset_description = {**dataset.data_description(), **dataset_description}

    stimulus_presentation = dataset_description.pop(
        "StimulusPresentation", {"SoftwareName": "unknown"}
    )
    descriptions = dataset.dataset[0].meta_data.get("_bids_descriptions", {})
    event_descriptions = {
        key.removeprefix("event:"): value
        for key, value in descriptions.items()
        if key.startswith("event:")
    }

    participant_runs = {}
    for index, data in enumerate(dataset.dataset):
        participant_id = _get_participant_id(data)
        if participant_id is None:
            participant_id = str(index)
        participant_id = _bids_label(participant_id)
        participant_runs[participant_id] = participant_runs.get(participant_id, 0) + 1
        _, filename, snirf_file = _bids_nirs_paths(
            path,
            participant_id,
            str(participant_runs[participant_id]),
            _get_task_name(data),
            _get_session_name(data),
        )
        legacy_file = snirf_file.with_name(
            filename + "_fnirs.snirf"
        )
        existing_file = snirf_file if snirf_file.exists() else legacy_file
        if existing_file.exists():
            _assert_snirf_raw_data_matches(existing_file, data)

    write_bids_readme(path=path, readme=readme)
    write_bids_dataset_description(path=path, description=dataset_description)

    add_missing_bids_to_metadata(dataset)

    if '_bids_descriptions' in dataset.dataset[0].meta_data:
        bids_dataset_description = dataset.dataset[0].meta_data['_bids_descriptions']
    else:
        bids_dataset_description = None

    demographics = dataset.get_demographics()
    participant_columns = {
        "participant_id", "subject_id", "subject", "subjid", "sub", "id", "name"
    }
    if not any(column.lower() in participant_columns for column in demographics.columns):
        demographics.insert(
            0,
            "subject",
            [
                _get_participant_id(data) or str(index)
                for index, data in enumerate(dataset.dataset)
            ],
        )

    save_bids_demographics(demographics=demographics,
                           path=path,
                           bids_descriptions=bids_dataset_description)

    all_participants = {}

    for i, data in enumerate(dataset.dataset):
        participant_id = _get_participant_id(data)
        if participant_id is None:
            participant_id = str(i)
        participant_id = _bids_label(participant_id)

        if participant_id not in all_participants:
            all_participants[participant_id] = 0
        all_participants[participant_id] += 1

        run = str(all_participants[participant_id])

        write_bids_data(data, path=path,
                        participant_id=participant_id,
                        run=run,
                        task=_get_task_name(data),
                        session=_get_session_name(data),
                        overwrite_snirf=overwrite_snirf,
                        include_derivatives=include_derivatives,
                        stimulus_presentation=stimulus_presentation,
                        event_descriptions=event_descriptions)


def _write_snirf_with_timeseries(path: Path, data, keys: list[str], stim: pd.DataFrame) -> None:
    missing_keys = [key for key in keys if key not in data.timeseries]
    if missing_keys:
        raise ValueError(f"Cannot write SNIRF file; missing timeseries key(s): {missing_keys}")
    original_timeseries = data.timeseries
    original_stim = data.stim
    data.timeseries = OrderedDict((key, original_timeseries[key]) for key in keys)
    data.stim = stim
    try:
        write_snirf(path, data)
    finally:
        data.timeseries = original_timeseries
        data.stim = original_stim


def _derivative_timeseries_keys(data) -> list[str]:
    return [key for key in getattr(data, "timeseries", {}) if key != "amp"]


def _derivative_snirf_path(root: Path, raw_snirf_file: Path) -> Path:
    relative = raw_snirf_file.relative_to(root)
    derivative_name = raw_snirf_file.with_name(f"{raw_snirf_file.stem}_derivatives.snirf").name
    return root / "bids_derivatives" / relative.with_name(derivative_name)


def _bids_label(value: str) -> str:
    """Convert a local identifier to the character set allowed for BIDS labels."""
    label = str(value)
    if label.startswith("sub-"):
        label = label[4:]
    label = re.sub(r"[^0-9A-Za-z+]", "+", label)
    if not label:
        raise ValueError("BIDS entity labels must contain at least one valid character")
    return label


def _migrate_legacy_bids_file(legacy_path: Path, bids_path: Path) -> None:
    """Move a legacy `_fnirs` file to its BIDS `_nirs` name without altering data."""
    if not legacy_path.exists():
        return
    if bids_path.exists():
        if legacy_path.read_bytes() != bids_path.read_bytes():
            raise FileExistsError(
                f"Both legacy and BIDS-named files exist with different contents: "
                f"{legacy_path} and {bids_path}"
            )
        legacy_path.unlink()
        return
    legacy_path.replace(bids_path)


def read_bids_dataset(folder: str | Path, include_derivatives: bool = True):
    """Read a NIRS-BIDS dataset folder into a :class:`DataSet`.

    Metadata stored in BIDS TSV/JSON files is applied after the SNIRF file is
    loaded, so sidecar files take precedence over the embedded SNIRF metadata.
    Standard ``*_events.tsv`` files override SNIRF stimulus data; legacy
    ``*_stim.tsv`` files are still read when no events file is present.
    Channels, optode geometry, and coordinate units are also restored from
    their BIDS sidecars when those files are available; absent sidecars leave
    the corresponding SNIRF values untouched.
    Matching sidecars are applied from the dataset root toward the local data
    folder, allowing more local files to override inherited parent files.
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"BIDS folder does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"BIDS path must be a folder: {root}")

    snirf_files = sorted(set(root.rglob("*_nirs.snirf")) | set(root.rglob("*_fnirs.snirf")))
    if not snirf_files:
        raise FileNotFoundError(f"No *_nirs.snirf or legacy *_fnirs.snirf files found in: {root}")

    dset = dataset.DataSet()
    description_path = root / "dataset_description.json"
    descriptors: dict[str, Any] = {
        "readme": _read_bids_readme(root / "README.md"),
        "dataset_description": {},
    }
    if description_path.exists():
        with open(description_path) as f:
            dataset_description = json.load(f)
        if isinstance(dataset_description, dict):
            descriptors["dataset_description"] = dataset_description.copy()
            descriptors["dataset_description"].setdefault(
                "StimulusPresentation", {"SoftwareName": "unknown"}
            )
        else:
            raise ValueError("dataset_description.json must contain a JSON object")
        dset.description = dataset_description.get("Name", dset.description)
    for snirf_file in snirf_files:
        loaded = read_bids_data(snirf_file, root, include_derivatives=include_derivatives)
        for rec in loaded:
            rec_descriptors = copy.deepcopy(descriptors)
            imported_descriptors = rec.meta_data.get("_bids_descriptors", {})
            imported_dataset_description = imported_descriptors.get(
                "dataset_description", {}
            )
            if isinstance(imported_dataset_description, dict):
                rec_descriptors["dataset_description"].update(
                    imported_dataset_description
                )
            rec.meta_data["_bids_descriptors"] = rec_descriptors
            dset.import_data(rec)

    return dset


def read_bids_data(
    snirf_file: str | Path,
    root: str | Path | None = None,
    include_derivatives: bool = True,
):
    snirf_file = Path(snirf_file)
    root = Path(root) if root is not None else _find_bids_root(snirf_file)
    loaded = read_snirf(snirf_file)
    for rec in loaded:
        _keep_timeseries(rec, ["amp"])
        if include_derivatives:
            _merge_bids_derivatives(rec, snirf_file, root)
        entities = _bids_entities_from_path(snirf_file, root)
        _apply_bids_participants_metadata(rec, snirf_file, root, entities)
        _apply_bids_fnirs_json_metadata(rec, snirf_file, root, entities)
        _apply_bids_stim_metadata(rec, snirf_file, root, entities)
        _apply_bids_events_json_metadata(rec, snirf_file, root, entities)
        _apply_bids_probe_metadata(rec, snirf_file, root, entities)
    return loaded


def _find_bids_root(path: Path) -> Path:
    for folder in [path.parent, *path.parents]:
        if (folder / "dataset_description.json").exists() or (folder / "participants.tsv").exists():
            return folder
    return path.parent


def _keep_timeseries(rec, keys: list[str]) -> None:
    rec.timeseries = OrderedDict(
        (key, value) for key, value in rec.timeseries.items() if key in keys
    )


def _merge_bids_derivatives(rec, snirf_file: Path, root: Path) -> None:
    derivative_file = _derivative_snirf_path(root, snirf_file)
    if not derivative_file.exists():
        return
    derivative_recordings = read_snirf(derivative_file)
    if not derivative_recordings:
        return
    for key, value in derivative_recordings[0].timeseries.items():
        if key != "amp":
            rec.timeseries[key] = value


def _bids_entities_from_path(path: Path, root: Path) -> dict:
    entities = {}
    for part in path.relative_to(root).parts[:-1]:
        if "-" in part:
            key, value = part.split("-", 1)
            if key in ("sub", "ses"):
                entities[key] = value
    entities.update(_bids_entities_from_name(path.name))
    return entities


def _bids_entities_from_name(name: str) -> dict:
    stem = Path(name).stem
    entities = {}
    for part in stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        entities[key] = value
    return entities


def _bids_sidecars_for_suffix(snirf_file: Path, root: Path, suffix: str, extension: str, entities: dict) -> list[Path]:
    sidecars = []
    for folder in _bids_ancestor_folders(snirf_file.parent, root):
        matches = []
        for candidate in folder.glob(f"*{extension}"):
            if not _bids_has_suffix(candidate, suffix, extension):
                continue
            candidate_entities = _bids_entities_from_name(candidate.name)
            if _bids_entities_match(candidate_entities, entities):
                matches.append(candidate)
        sidecars.extend(sorted(matches, key=lambda path: len(_bids_entities_from_name(path.name))))
    return sidecars


def _bids_ancestor_folders(folder: Path, root: Path) -> list[Path]:
    relative_parts = folder.relative_to(root).parts
    folders = [root]
    current = root
    for part in relative_parts:
        current = current / part
        folders.append(current)
    return folders


def _bids_has_suffix(path: Path, suffix: str, extension: str) -> bool:
    name = path.name
    return name == f"{suffix}{extension}" or name.endswith(f"_{suffix}{extension}")


def _bids_entities_match(candidate_entities: dict, target_entities: dict) -> bool:
    for key, value in candidate_entities.items():
        if key in target_entities and str(target_entities[key]) == str(value):
            continue
        return False
    return True


def _apply_bids_fnirs_json_metadata(rec, snirf_file: Path, root: Path, entities: dict) -> None:
    for suffix in ("fnirs", "nirs"):
        for sidecar in _bids_sidecars_for_suffix(snirf_file, root, suffix, ".json", entities):
            with open(sidecar) as f:
                rec.meta_data.update(json.load(f))


def _apply_bids_participants_metadata(rec, snirf_file: Path, root: Path, entities: dict) -> None:
    participant_id = entities.get("sub")
    if participant_id is None:
        return

    participants_tsv = None
    for candidate in _bids_ancestor_folders(snirf_file.parent, root):
        path = candidate / "participants.tsv"
        if path.exists():
            participants_tsv = path
    if participants_tsv is None:
        return

    participants = pd.read_csv(participants_tsv, sep="\t", keep_default_na=False)
    row = _bids_participant_row(participants, participant_id)
    if row is None:
        return

    participants_json = _bids_participants_json(snirf_file, root)
    for key, value in row.items():
        if value == "":
            continue
        rec.meta_data[key] = _apply_bids_participant_unit(value, participants_json.get(key, {}))

    descriptions = {
        key: value.get("Description")
        for key, value in participants_json.items()
        if isinstance(value, dict) and value.get("Description") is not None
    }
    if descriptions:
        rec.meta_data.setdefault("_bids_descriptions", {}).update(descriptions)


def _bids_participants_json(snirf_file: Path, root: Path) -> dict:
    participants_json = {}
    for candidate in _bids_ancestor_folders(snirf_file.parent, root):
        path = candidate / "participants.json"
        if path.exists():
            with open(path) as f:
                participants_json.update(json.load(f))
    return participants_json


def _bids_participant_row(participants: pd.DataFrame, participant_id: str):
    participant_columns = [
        'participant_id', 'subject_id', 'subject', 'subjid', 'sub', 'id', 'name'
    ]
    for column in participants.columns:
        if column.lower() not in participant_columns:
            continue
        for _, row in participants.iterrows():
            value = str(row[column])
            if value == participant_id or value == f"sub-{participant_id}":
                return row
    if len(participants) == 1:
        return participants.iloc[0]
    return None


def _apply_bids_participant_unit(value, column_info: dict):
    unit_name = column_info.get("Units") if isinstance(column_info, dict) else None
    if unit_name in (None, "", "n/a"):
        return value
    if not dataset.DataSet._is_demographics_unit_value(value):
        return value
    return value * units.Unit(unit_name)


def _apply_bids_stim_metadata(rec, snirf_file: Path, root: Path, entities: dict) -> None:
    stim_tsv_files = _bids_sidecars_for_suffix(snirf_file, root, "events", ".tsv", entities)
    if not stim_tsv_files:
        stim_tsv_files = _bids_sidecars_for_suffix(snirf_file, root, "stim", ".tsv", entities)
    if not stim_tsv_files:
        return
    stim = pd.read_csv(stim_tsv_files[-1], sep="\t", keep_default_na=False)
    rename = {}
    if "trial_type" not in stim and "name" in stim:
        rename["name"] = "trial_type"
    if "value" not in stim and "amplitude" in stim:
        rename["amplitude"] = "value"
    stim.rename(columns=rename, inplace=True)
    rec.stim = stim


def _apply_bids_events_json_metadata(
    rec, snirf_file: Path, root: Path, entities: dict
) -> None:
    event_sidecars = _bids_sidecars_for_suffix(
        snirf_file, root, "events", ".json", entities
    )
    if not event_sidecars:
        return

    metadata = {}
    for sidecar in event_sidecars:
        with open(sidecar) as f:
            current = json.load(f)
        if not isinstance(current, dict):
            raise ValueError(f"Events JSON sidecar must contain an object: {sidecar}")
        for key, value in current.items():
            previous = metadata.get(key)
            if isinstance(previous, dict) and isinstance(value, dict):
                merged = previous.copy()
                for field, field_value in value.items():
                    if (
                        field == "Levels"
                        and isinstance(merged.get(field), dict)
                        and isinstance(field_value, dict)
                    ):
                        merged[field] = {**merged[field], **field_value}
                    else:
                        merged[field] = field_value
                metadata[key] = merged
            else:
                metadata[key] = value

    trial_type = metadata.get("trial_type")
    levels = trial_type.get("Levels") if isinstance(trial_type, dict) else None
    if isinstance(levels, dict):
        descriptions = rec.meta_data.setdefault("_bids_descriptions", {})
        descriptions.update({
            f"event:{event_type}": description
            for event_type, description in levels.items()
            if isinstance(event_type, str) and isinstance(description, str)
        })

    stimulus_presentation = metadata.get("StimulusPresentation")
    if stimulus_presentation is not None:
        rec.meta_data.setdefault("_bids_descriptors", {}).setdefault(
            "dataset_description", {}
        )["StimulusPresentation"] = stimulus_presentation


def _apply_bids_probe_metadata(
    rec, snirf_file: Path, root: Path, entities: dict
) -> None:
    channel_files = _bids_sidecars_for_suffix(
        snirf_file, root, "channels", ".tsv", entities
    )
    if channel_files:
        channels = pd.read_csv(channel_files[-1], sep="\t", keep_default_na=False)
        _apply_bids_channels_tsv(rec, channels)

    optode_files = _bids_sidecars_for_suffix(
        snirf_file, root, "optodes", ".tsv", entities
    )
    if not optode_files:
        return

    coordinate_system = {}
    for sidecar in _bids_sidecars_for_suffix(
        snirf_file, root, "coordsystem", ".json", entities
    ):
        with open(sidecar) as f:
            values = json.load(f)
        if not isinstance(values, dict):
            raise ValueError(f"Coordinate-system sidecar must contain an object: {sidecar}")
        coordinate_system.update(values)

    optodes = pd.read_csv(optode_files[-1], sep="\t", keep_default_na=False)
    _apply_bids_optodes_tsv(rec, optodes, coordinate_system)


def _apply_bids_channels_tsv(rec, channels: pd.DataFrame) -> None:
    required = {"name", "source", "detector", "wavelength_nominal"}
    missing = required.difference(channels.columns)
    if missing:
        raise ValueError(
            f"BIDS channels.tsv is missing required column(s): {sorted(missing)}"
        )
    if "amp" not in getattr(rec, "timeseries", {}):
        return

    ts = rec["amp"]
    if not {"channel", "wavelength"}.issubset(ts.dims):
        raise ValueError("SNIRF amplitude timeseries must have channel and wavelength dimensions")

    channel_names = list(pd.unique(channels["name"].astype(str)))
    wavelengths = list(
        pd.unique(pd.to_numeric(channels["wavelength_nominal"], errors="raise"))
    )
    expected_rows = len(channel_names) * len(wavelengths)
    if (
        len(channels) != expected_rows
        or len(channel_names) != ts.sizes["channel"]
        or len(wavelengths) != ts.sizes["wavelength"]
    ):
        raise ValueError(
            "BIDS channels.tsv channel/wavelength rows do not match the SNIRF "
            "amplitude dimensions"
        )

    sources = []
    detectors = []
    for channel_name in channel_names:
        rows = channels[channels["name"].astype(str) == channel_name]
        if rows["source"].nunique() != 1 or rows["detector"].nunique() != 1:
            raise ValueError(
                f"BIDS channels.tsv has inconsistent source/detector for {channel_name}"
            )
        if set(pd.to_numeric(rows["wavelength_nominal"], errors="raise")) != set(
            wavelengths
        ):
            raise ValueError(
                f"BIDS channels.tsv has incomplete wavelengths for {channel_name}"
            )
        sources.append(str(rows.iloc[0]["source"]))
        detectors.append(str(rows.iloc[0]["detector"]))

    ts = ts.assign_coords(
        channel=("channel", channel_names),
        source=("channel", sources),
        detector=("channel", detectors),
        wavelength=("wavelength", wavelengths),
    )
    if "units" in channels:
        channel_units = list(pd.unique(channels["units"].astype(str)))
        if len(channel_units) != 1:
            raise ValueError("BIDS channels.tsv contains inconsistent channel units")
        unit_name = channel_units[0]
        if unit_name and unit_name != "n/a":
            ts = ts.copy(data=ts.pint.magnitude * units.Unit(unit_name))
    rec["amp"] = ts


def _apply_bids_optodes_tsv(
    rec, optodes: pd.DataFrame, coordinate_system: dict
) -> None:
    required = {"name", "type", "x", "y"}
    missing = required.difference(optodes.columns)
    if missing:
        raise ValueError(
            f"BIDS optodes.tsv is missing required column(s): {sorted(missing)}"
        )

    geo = getattr(rec, "geo3d", None)
    if geo is None:
        return
    if "label" not in geo.dims or "pos" not in geo.dims:
        raise ValueError("SNIRF probe geometry must have label and pos dimensions")
    if len(optodes) != geo.sizes["label"]:
        raise ValueError("BIDS optodes.tsv row count does not match SNIRF probe geometry")

    position_columns = ["x", "y"]
    if geo.sizes["pos"] == 3:
        if "z" not in optodes:
            raise ValueError("BIDS optodes.tsv is missing the z coordinate")
        position_columns.append("z")
    elif geo.sizes["pos"] != 2:
        raise ValueError("SNIRF probe geometry must have two or three position dimensions")

    position_values = optodes[position_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(position_values).all():
        raise ValueError("BIDS optodes.tsv contains missing or invalid coordinates")

    coordinate_unit = coordinate_system.get("NIRSCoordinateUnits", "mm")
    positions = (
        position_values * units.Unit(coordinate_unit)
    ).to(units.millimeter).magnitude
    point_types = []
    for value in optodes["type"].astype(str):
        if value.lower() == "source":
            point_types.append(PointType.SOURCE)
        elif value.lower() == "detector":
            point_types.append(PointType.DETECTOR)
        else:
            raise ValueError(f"Unsupported BIDS optode type: {value}")

    geo = geo.pint.to(units.millimeter).copy(
        data=positions * units.millimeter
    )
    rec.geo3d = geo.assign_coords(
        label=("label", optodes["name"].astype(str).tolist()),
        type=("label", point_types),
    )


def write_bids_stim(
    stim: pd.DataFrame,
    path: str | Path,
    stimulus_presentation: dict[str, Any] | None = None,
    event_descriptions: dict[str, str] | None = None,
):
    """Write stimulus information using the BIDS events.tsv schema."""
    stim = stim.copy()
    rename = {}
    if "trial_type" not in stim and "name" in stim:
        rename["name"] = "trial_type"
    if "value" not in stim and "amplitude" in stim:
        rename["amplitude"] = "value"
    stim.rename(columns=rename, inplace=True)
    required_columns = ["onset", "duration"]
    missing_columns = [column for column in required_columns if column not in stim]
    if missing_columns:
        raise ValueError(f"BIDS events data is missing required column(s): {missing_columns}")

    ordered_columns = required_columns
    if "trial_type" in stim:
        ordered_columns = [*ordered_columns, "trial_type"]
    ordered_columns.extend(column for column in stim.columns if column not in ordered_columns)
    stim = stim[ordered_columns]

    info = {
        "Description": (
            "Events recorded for this task. "
            + (
                " ".join(
                    f"{event_type}: {description}"
                    for event_type, description in event_descriptions.items()
                )
                if event_descriptions
                else "The table contains event onset, duration, and condition information."
            )
        ),
        "onset": {"Description": "Onset of the event.", "Units": "s"},
        "duration": {"Description": "Duration of the event.", "Units": "s"},
    }
    if "trial_type" in stim:
        trial_type_info = {
            "Description": "Task condition or event type associated with the event.",
        }
        if event_descriptions:
            event_types = {str(value) for value in stim["trial_type"].dropna().unique()}
            levels = {
                str(eventkey): description
                for eventkey, description in event_descriptions.items()
                if str(eventkey) in event_types
            }
            if levels:
                trial_type_info["Levels"] = levels
        info["trial_type"] = trial_type_info
    if "value" in stim:
        info["value"] = {"Description": "Amplitude of the event."}
    for column in stim.columns:
        info.setdefault(column, {"Description": "Additional event information."})

    stim.to_csv(Path(path).with_suffix('.tsv'), sep='\t', index=False, na_rep="n/a")

    stimfile_json = Path(path).with_suffix(".json")
    info["StimulusPresentation"] = (
        {"SoftwareName": "unknown"}
        if stimulus_presentation is None
        else stimulus_presentation
    )
    with open(stimfile_json, "w") as f:
        json.dump(info, f, indent=4)


def _get_participant_id(data):
    keys = ['subject_id', 'subject', 'subjid', 'participant_id', 'sub', 'id', 'name']
    return _get_meta_data_value_case_insensitive(data, keys, None)


def _get_task_name(data):
    keys = ['task', 'taskname']
    return _get_meta_data_value_case_insensitive(data, keys, 'func')


def _get_session_name(data):
    keys = ['session', 'session_id', 'ses']
    return _get_meta_data_value_case_insensitive(data, keys, None)


def _get_meta_data_value_case_insensitive(data, keys, default):
    meta_data = getattr(data, "meta_data", {})
    lower_to_key = {key.lower(): key for key in meta_data}
    for key in keys:
        actual_key = lower_to_key.get(key.lower())
        if actual_key is not None:
            return meta_data[actual_key]
    return default


def bids_README_template():
    """Return a template for the BIDS README file."""

    # https://bids.neuroimaging.io/getting_started/templates/index.html

    bids_README_template = {}
    bids_README_template['Overview'] = (
        "State the scientific question / purpose, then summarize the essentials — modality, "
        "number of participants and any groups, sessions/runs, task(s), approximate "
        "recording duration, the project name and years it ran, and the associated "
        "publication (if any)"
    )
    bids_README_template['Experimental design'] = (
        "State the variables that define the experiment — the independent (manipulated), "
        "dependent (measured), and control (held-fixed) variables. This does not apply to "
        "datasets without a designed manipulation (e.g. resting state)."
    )
    bids_README_template['Dataset contents and structure'] = (
        "Orient the reader to what they cannot infer from the standard BIDS layout, for "
        "example contents of derivatives/ and sourcedata/, extra or non-standard files, "
        "and any intentional deviation from BIDS."
    )
    bids_README_template['Methods'] = (
        "If the dataset has an associated paper, you may copy or adapt its Methods "
        "section here, then trim it to what is relevant to the data as shared."
    )
    bids_README_template['Participants and recruitment'] = (
        "Recruitment, eligibility, grouping, and how many were excluded and why. Do "
        "NOT paste per-subject demographics — those belong in participants.tsv, and "
        "Control/Patient status belongs in its `group` column. Only summarize."
    )
    bids_README_template['Acquisition'] = (
        "Short description of the equipment and environment (e.g. shielded room, "
        "seated/supine for MEG, any setup done when the subject arrived), plus the "
        "few parameters needed to understand the signals. Full machine-readable "
        "parameters belong in *_<modality>.json."
    )
    bids_README_template['Task and paradigm'] = (
        "What participants did and the trial structure, plus how tasks were organized "
        "across a session (order, counter-balancing, activities between tasks). The "
        "canonical TaskName / TaskDescription live in task-<label>_*.json and events "
        "live in *_events.tsv"
    )
    bids_README_template['Stimuli'] = (
        "What was presented, and where stimulus files live (e.g. stimuli/)."
    )
    bids_README_template['Additional data acquired'] = (
        "Non-imaging data collected as part of the study: questionnaires, surveys, "
        "clinical measures, swabs. Note availability and location; standardized "
        "phenotypic data belong in a phenotype/ folder."
    )
    bids_README_template['Known issues, quality, and missing data'] = (
        "This is the highest-value section for data reuse. Give a short overall "
        "quality summary (with a link to e.g. an MRIQC report if available), then "
        "anything that affects analysis: missing/partial runs, excluded participants, "
        "bad channels, timing or trigger problems, equipment or protocol changes "
        "mid-study,  a lesion or anomaly in one participant, or data that look "
        "normal but are not. Write \"None known.\" if the dataset is clean."
    )
    bids_README_template['How to use these data'] = (
        "Preprocessing already applied (and where it lives), recommended "
        "reference/montage, software + version, or analysis tips."
    )
    bids_README_template['References and citation'] = (
        "A one-line \"how to cite\" and the key paper(s). The dataset DOI and "
        "machine-readable references also live in dataset_description.json"
    )
    bids_README_template['Contact'] = (
        "The person or team that can give additional information. "
        "A role + email + ORCID is more future-proof than a personal name alone."
    )

    return bids_README_template


def bids_dataset_description_template():
    """Return a template for the BIDS dataset_description.json file."""
    bids_dataset_description_template = {}
    bids_dataset_description_template['Name'] = "Dataset Name"
    bids_dataset_description_template['BIDSVersion'] = "1.10.0"
    bids_dataset_description_template['License'] = "CC0"
    bids_dataset_description_template['Authors'] = ["Author 1", "Author 2"]
    bids_dataset_description_template['Acknowledgements'] = "Acknowledgements"
    bids_dataset_description_template['HowToAcknowledge'] = "How to acknowledge"
    bids_dataset_description_template['Funding'] = ["Funding source 1", "Funding source 2"]
    bids_dataset_description_template['ReferencesAndLinks'] = ["Reference 1", "Reference 2"]
    bids_dataset_description_template['GeneratedBy'] = [{"Name": "unknown"}]
    bids_dataset_description_template['StimulusPresentation'] = {
        "SoftwareName": "unknown"
    }

    return bids_dataset_description_template


def write_bids_readme(path: str | Path, readme: dict):
    """Write the dataset_description.json file in the BIDS dataset directory.

    Args:
        path (str | Path): The directory path where the dataset_description.json file will be saved.
        readme (dict): The content to be written in the README.md file.
    """

    key_order = [
        'Overview',
        'Experimental design ',
        'Dataset contents and structure',
        'Methods',
        'Participants and recruitment',
        'Acquisition',
        'Task and paradigm',
        'Stimuli ',
        'Additional data acquired ',
        'Known issues, quality, and missing data',
        'How to use these data',
        'References and citation',
        'Contact']

    added_keys = [key for key in readme.keys() if key not in key_order]
    key_order.extend(added_keys)

    readme_path = Path(path) / 'README.md'
    readme_path.parent.mkdir(parents=True, exist_ok=True)

    with open(readme_path, 'w') as f:
        for key in key_order:
            if key in readme:
                f.write(f"### {key}\n")
                f.write(f"{readme[key]}\n\n")


def _read_bids_readme(path: Path) -> dict[str, str]:
    """Read section headings and bodies from a BIDS README markdown file."""
    if not path.exists():
        return {}

    sections: dict[str, list[str]] = {}
    preamble: list[str] = []
    current_section = None
    with open(path) as f:
        for line in f:
            heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line.rstrip())
            if heading:
                current_section = heading.group(1)
                sections.setdefault(current_section, [])
            elif current_section is not None:
                sections[current_section].append(line.rstrip())
            else:
                preamble.append(line.rstrip())

    result = {title: "\n".join(body).strip() for title, body in sections.items()}
    preamble_text = "\n".join(preamble).strip()
    if preamble_text:
        result["Overview"] = (
            f"{preamble_text}\n\n{result['Overview']}"
            if result.get("Overview")
            else preamble_text
        )
    return result


def write_bids_dataset_description(path: str | Path, description: dict):
    """Write a dataset_description.json file in the BIDS dataset directory.

    Args:
        path (str | Path): The directory path where the dataset_description.json file will be saved.
        description (dict): The content to be written in the dataset_description.json file.
    """
    description_path = Path(path) / 'dataset_description.json'
    description_path.parent.mkdir(parents=True, exist_ok=True)

    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)


def add_missing_bids_to_metadata(data, intype='amp'):
    """Add missing BIDS metadata to the existing metadata dictionary.

    Args:
        data: The data object containing metadata and other relevant information.
        intype (str): The type of data to be processed (default is 'amp').
    """

    # The NIRS-BIDS validator requires certain fields to be present in the metadata,
    # even if they are not strictly necessary for the BIDS specification or the SNIRF
    # standard. This function adds those required fields with default values if missing.

    if (data.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
        for r in data.dataset:
            add_missing_bids_to_metadata(r, intype=intype)
        return
    else:
        defaults = {
            'Manufacturer': {
                'value': 'n/a',
                'description': 'The manufacturer of the device used for data      acquisition',
                'units': 'n/a',
            },
            'ManufacturersModelName': {
                'value': 'n/a',
                'description': 'The model name of the device used for data acquisition.',
                'units': 'n/a',
            },
            'SoftwareVersions': {
                'value': 'n/a',
                'description': 'The software versions of the device used for data acquisition',
                'units': 'n/a',
            },
            'DeviceSerialNumber': {
                'value': 'n/a',
                'description': 'The serial number of the device used for data acquisition.',
                'units': 'n/a',
            },
            'HardwareFilters': {
                'value': 'n/a',
                'description': 'The hardware filters used in the device.',
                'units': 'n/a',
            },
            'SourceType': {
                'value': 'n/a',
                'description': 'The type of source used in the device.',
                'units': 'n/a',
            },
            'DetectorType': {
                'value': 'n/a',
                'description': 'The type of detector used in the device.',
                'units': 'n/a',
            },
            'InstitutionName': {
                'value': 'n/a',
                'description': 'The name of the institution where the data was acquired.',
                'units': 'n/a',
            },
            'InstitutionAddress': {
                'value': 'n/a',
                'description': 'The address of the institution where the data was acquired.',
                'units': 'n/a',
            },
            'InstitutionalDepartmentName': {
                'value': 'n/a',
                'description': 'The department of the institution where the data was acquired.',
                'units': 'n/a',
            },
            'CapManufacturer': {
                'value': 'n/a',
                'description': 'The manufacturer of the cap used in the study.',
                'units': 'n/a',
            },
            'CapManufacturersModelName': {
                'value': 'n/a',
                'description': 'The model name of the cap used in the study.',
                'units': 'n/a',
            },
            'HeadCircumference': {
                'value': 'n/a',
                'description': 'The head circumference of the subject.',
                'units': units.centimeter,
            },
            'SubjectArtefactDescription': {
                'value': 'n/a',
                'description': 'Description of any artefacts related to the subject.',
                'units': 'n/a',
            },
            'TaskDescription': {
                'value': 'n/a',
                'description': 'Description of the task performed by the subject.',
                'units': 'n/a',
            },
            'Instructions': {
                'value': 'n/a',
                'description': 'Instructions given to the subject.',
                'units': 'n/a',
            },
            'CogPOID': {
                'value': 'n/a',
                'description': 'The CogPO ID associated with the task.',
                'units': 'n/a',
            },
            'NIRSPlacementScheme': {
                'value': 'n/a',
                'description': 'The NIRS placement scheme used.',
                'units': 'n/a',
            },
            'RecordingDuration': {
                'value': (data[intype].time[-1] - data[intype].time[0]).values,
                'description': 'The duration of the recording.',
                'units': units.seconds,
            },
            'SamplingFrequency': {
                'value': 1 / (data[intype].time[1] - data[intype].time[0]).values,
                'description': 'The sampling frequency of the recording.',
                'units': units.hertz,
            },
            'NIRSChannelCount': {
                'value': (
                    len(data[intype].channel)
                    * len(data[intype].wavelength)
                ),
                'description': 'The number of NIRS channels.',
                'units': 'n/a',
            },
            'NIRSSourceOptodeCount': {
                'value': sum((data.geo2d.type == PointType(1).SOURCE).values),
                'description': 'The number of NIRS source optodes.',
                'units': 'n/a',
            },
            'NIRSDetectorOptodeCount': {
                'value': sum((data.geo2d.type == PointType(1).DETECTOR).values),
                'description': 'The number of NIRS detector optodes.',
                'units': 'n/a',
            },
            'ShortChannelCount': {
                'value': sum(
                    (
                        xrutils.norm(
                            data.geo3d.loc[data[intype].source]
                            - data.geo3d.loc[data[intype].detector],
                            data.geo3d.points.crs,
                        )
                        < units.millimeter * 15
                    ).values
                ),
                'description': 'The number of short channels.',
                'units': 'n/a',
            },
        }

        for key, value in defaults.items():
            if key not in data.meta_data:
                data.meta_data[key] = value['value']
                if '_bids_descriptions' not in data.meta_data:
                    data.meta_data['_bids_descriptions'] = {}

                if key not in data.meta_data['_bids_descriptions']:
                    data.meta_data['_bids_descriptions'][key] = value['description']

            if value['units'] != 'n/a':
                # Add the units information to the data.meta_data
                if key in data.meta_data:
                    magnitude, _ = dataset.DataSet._split_demographics_units(data.meta_data[key])
                    if dataset.DataSet._is_demographics_unit_value(magnitude):
                        data.meta_data[key] = magnitude * value['units']

        return


def _strip_units_from_demographics(demographics: pd.DataFrame) -> pd.DataFrame:
    stripped = demographics.copy()
    for col in stripped.columns:
        stripped[col] = stripped[col].map(_strip_units_from_demographics_value)
    return stripped


def _strip_units_from_demographics_value(value):
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        magnitude, _ = dataset.DataSet._split_demographics_units(value)
        return magnitude
    return value


def save_bids_demographics(demographics: pd.DataFrame, path: str | Path, bids_descriptions: dict = None):
    """Save the demographics dataframe in BIDS format.

    Args:
        demographics (pd.DataFrame): The demographics dataframe to be saved.
        demofile (str | Path): The file path where the demographics dataframe will be saved.
    """
    if not isinstance(demographics, pd.DataFrame):
        raise ValueError("demographics must be a pandas DataFrame")

    if not isinstance(path, (str, Path)):
        raise ValueError("path must be a string or Path")

    participant_columns = {
        'participant_id', 'subject_id', 'subject', 'subjid', 'sub', 'id', 'name'
    }
    recording_columns = {
        "task", "taskname", "session", "session_id", "ses", "run", "run_id"
    }
    id_column = next(
        (column for column in demographics.columns if column.lower() in participant_columns),
        None,
    )
    if id_column is None:
        raise ValueError("demographics must include a participant or subject ID column")

    participant_demographics = pd.DataFrame(index=demographics.index)
    participant_demographics['participant_id'] = demographics[id_column].map(
        lambda value: f"sub-{_bids_label(value)}"
    )
    description_columns = {'participant_id': id_column}
    for column in demographics.columns:
        if (
            column == id_column
            or column.lower() in participant_columns
            or column.lower() in recording_columns
        ):
            continue
        participant_demographics[column] = demographics[column]
        description_columns[column] = column

    participant_demographics = participant_demographics.reset_index(drop=True)
    duplicate_ids = participant_demographics["participant_id"].duplicated(keep=False)
    if duplicate_ids.any():
        repeated = participant_demographics.loc[duplicate_ids]
        conflicts = [
            column
            for column in participant_demographics.columns
            if column != "participant_id"
            and repeated.groupby("participant_id", dropna=False)[column]
            .agg(lambda values: values.astype(str).nunique())
            .gt(1)
            .any()
        ]
        if conflicts:
            raise ValueError(
                "Repeated participants have conflicting demographic values in "
                f"column(s): {conflicts}"
            )
        participant_demographics = participant_demographics.drop_duplicates(
            subset="participant_id", keep="first"
        )

    demofileTSV = Path(path) / 'participants.tsv'
    demofileTSV.parent.mkdir(parents=True, exist_ok=True)
    _strip_units_from_demographics(participant_demographics).to_csv(
        demofileTSV, sep='\t', index=False, na_rep="n/a"
    )

    # Now save the JSON sidecar file with the column descriptions
    demofileJSON = Path(path) / 'participants.json'
    demofileJSON.parent.mkdir(parents=True, exist_ok=True)

    demo_info = {}

    for col, source_column in description_columns.items():
        description = (
            "Unique BIDS participant identifier."
            if col == "participant_id"
            else "No description available."
        )
        if bids_descriptions is not None:
            description = bids_descriptions.get(source_column, bids_descriptions.get(col, description))

        first_value = demographics[source_column].iloc[0] if not demographics.empty else None
        if type(first_value) == units.Quantity:  # noqa: E721
            demo_info[col] = {
                "Description": description,
                "Units": str(first_value.units)
            }
        else:
            demo_info[col] = {"Description": description}

    with open(demofileJSON, 'w') as f:
        json.dump(demo_info, f, indent=4)
