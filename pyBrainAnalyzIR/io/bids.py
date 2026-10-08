from pathlib import Path
import pandas as pd
import json
from collections import OrderedDict
from cedalion.dataclasses.geometry import PointType
import cedalion.xrutils as xrutils

from cedalion.io.snirf import read_snirf, write_snirf
from datetime import datetime

from cedalion import units
import pyBrainAnalyzIR
import pyBrainAnalyzIR.dataclasses.dataset as dataset


"""# BIDS Dataset
This dataset is organized according to the Brain Imaging Data Structure (BIDS) standard. For more information about BIDS, please visit [https://bids.neuroimaging.io/](https://bids.neuroimaging.io/).  # noqa: E501
"""

def write_bids_data(data, path: str | Path,
                    participant_id: str,
                    run: str,
                    task: str = 'func',
                    session: str | None = None,
                    overwrite_snirf: bool = False,
                    include_derivatives: bool = True):
    """Write individual data to a BIDS-compliant directory."""
    if session is None:
        folder = Path(path) / f"sub-{participant_id}"
        filename = f"sub-{participant_id}_task-{task}_run-{run}"
    else:
        folder = Path(path) / f"sub-{participant_id}" / f"ses-{session}"
        filename = f"sub-{participant_id}_ses-{session}_task-{task}_run-{run}"

    snirffile = folder / (filename + "_fnirs.snirf")
    snirffile.parent.mkdir(parents=True, exist_ok=True)

    # TODO - the SNIRF writer assumes this order.  It shouldn't assume, but it isn't my code so I just need to conform to it.
    stim = data.stim[["trial_type","onset", "duration", "value"]]

    if overwrite_snirf or not snirffile.exists():
        _write_snirf_with_timeseries(snirffile, data, ["amp"], stim)
    write_bids_stim(stim.copy(), path=folder / (filename + "_events.tsv"))

    derivative_keys = _derivative_timeseries_keys(data)
    if include_derivatives and derivative_keys:
        derivative_file = _derivative_snirf_path(Path(path), snirffile)
        derivative_file.parent.mkdir(parents=True, exist_ok=True)
        if overwrite_snirf or not derivative_file.exists():
            _write_snirf_with_timeseries(derivative_file, data, derivative_keys, stim)

    info={'TaskName': task,
          'Version': data.meta_data.get('Version', 'unknown'),
          'MeasurementDate': data.meta_data.get('MeasurementDate', 'unknown'),
          'MeasurementTime': data.meta_data.get('MeasurementTime', 'unknown'),
          'SNIRF_createDate': data.meta_data.get('SNIRF_CreateDate', datetime.now().strftime('%Y-%m-%d')),
          'SNIRF_createTime': data.meta_data.get('SNIRF_createTime', f"{datetime.now().strftime('%H:%M:%S')}Z"),
          'filedescription': data.meta_data.get('filedescription', 'unknown')
    }
    snirffilejson = folder / (filename + "_fnirs.json")
    with open(snirffilejson, 'w') as f:
        json.dump(info, f, indent=4)



def write_bids_dataset(dataset, path: str | Path,
                       readme: dict | None = None,
                       dataset_description: dict | None = None,
                       overwrite_snirf: bool = False,
                       include_derivatives: bool = True):

    """Write the dataset to a BIDS-compliant directory."""

    if readme is None:
        readme = bids_README_template()
    if dataset_description is None:
        dataset_description = bids_dataset_description_template()


    write_bids_readme(path=path, readme=readme)
    write_bids_dataset_description(path=path, description=dataset_description)

    add_missing_bids_to_metadata(dataset)

    if '_bids_descriptions' in dataset.dataset[0].meta_data:
        bids_dataset_description = dataset.dataset[0].meta_data['_bids_descriptions']
    else:
        bids_dataset_description = None

    save_bids_demographics(demographics=dataset.get_demographics(),
                           path=path,
                           bids_descriptions=bids_dataset_description)

    all_participants = {}

    for i, data in enumerate(dataset.dataset):
        participant_id = _get_participant_id(data)
        if participant_id is None:
            participant_id = str(i)
            
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
                    include_derivatives=include_derivatives)


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


def read_bids_dataset(folder: str | Path, include_derivatives: bool = True):
    """Read a NIRS-BIDS dataset folder into a :class:`DataSet`.

    Metadata stored in BIDS TSV/JSON files is applied after the SNIRF file is
    loaded, so sidecar files take precedence over the embedded SNIRF metadata.
    Standard ``*_events.tsv`` files override SNIRF stimulus data; legacy
    ``*_stim.tsv`` files are still read when no events file is present.
    Matching sidecars are applied from the dataset root toward the local data
    folder, allowing more local files to override inherited parent files.
    """
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"BIDS folder does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"BIDS path must be a folder: {root}")

    snirf_files = sorted(root.rglob("*_fnirs.snirf"))
    if not snirf_files:
        raise FileNotFoundError(f"No *_fnirs.snirf files found in: {root}")

    dset = dataset.DataSet()
    description_path = root / "dataset_description.json"
    if description_path.exists():
        with open(description_path) as f:
            dataset_description = json.load(f)
        dset.description = dataset_description.get("Name", dset.description)

    for snirf_file in snirf_files:
        loaded = read_bids_data(snirf_file, root, include_derivatives=include_derivatives)
        for rec in loaded:
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
        _apply_bids_fnirs_json_metadata(rec, snirf_file, root, entities)
        _apply_bids_participants_metadata(rec, snirf_file, root, entities)
        _apply_bids_stim_metadata(rec, snirf_file, root, entities)
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
    for sidecar in _bids_sidecars_for_suffix(snirf_file, root, "fnirs", ".json", entities):
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
    participant_columns = ['participant_id', 'subject_id', 'subject', 'subjid', 'sub','id','name']
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


def write_bids_stim(stim: pd.DataFrame, path: str | Path):
    """Write stimulus information using the BIDS events.tsv schema."""
    stim = stim[["trial_type", "onset", "duration", "value"]]

    info = {
        "trial_type": {"Description": "Type or name of the event."},
        "onset": {"Description": "Onset of the event.", "Units": "seconds"},
        "duration": {"Description": "Duration of the event.", "Units": "seconds"},
        "value": {"Description": "Amplitude of the event."},
    }

    stim.to_csv(Path(path).with_suffix('.tsv'), sep='\t', index=False)
    
    stimfile_json = Path(path).with_suffix(".json")
    with open(stimfile_json, "w") as f:
        json.dump(info, f, indent=4)


def _get_participant_id(data):
    keys = ['subject_id', 'subject', 'subjid', 'participant_id', 'sub','id','name']
    return _get_meta_data_value_case_insensitive(data, keys, None)

def _get_task_name(data):
    keys = ['task','taskname']
    return _get_meta_data_value_case_insensitive(data, keys, 'func')


def _get_session_name(data):
    keys = ['session','session_id','ses']
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
    bids_dataset_description_template['How To Acknowledge'] = "How to acknowledge"
    bids_dataset_description_template['Funding'] = ["Funding source 1", "Funding source 2"]
    bids_dataset_description_template['ReferencesAndLinks'] = ["Reference 1", "Reference 2"]
    bids_dataset_description_template['DatasetDOI'] = "doi"

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

    # The NIRS-BIDS validator requires certain fields to be present in the metadata, even if they are not strictly necessary for the BIDS specification or the SNIRF standard. This function adds those required fields with default values if they are missing.

    if (data.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
        for r in data.dataset:
            add_missing_bids_to_metadata(r, intype=intype)
        return
    else:
        defaults = {'Manufacturer': {'value':'n/a','description':'The manufacturer of the device used for data      acquisition','units': 'n/a'},
                    'ManufacturersModelName': {'value':'n/a','description':'The model name of the device used for data acquisition.','units': 'n/a'},
                    'SoftwareVersions': {'value':'n/a','description':'The software versions of the device used for data acquisition','units': 'n/a'},
                    'DeviceSerialNumber': {'value':'n/a','description':'The serial number of the device used for data acquisition.','units': 'n/a'},
                    'HardwareFilters': {'value':'n/a','description':'The hardware filters used in the device.','units': 'n/a'},
                    'SourceType': {'value':'n/a','description':'The type of source used in the device.','units': 'n/a'},
                    'DetectorType': {'value':'n/a','description':'The type of detector used in the device.','units': 'n/a'},
                    'InstitutionName': {'value':'n/a','description':'The name of the institution where the data was acquired.','units': 'n/a'},
                    'InstitutionAddress': {'value':'n/a','description':'The address of the institution where the data was acquired.','units': 'n/a'},
                    'InstitutionalDepartmentName': {'value':'n/a','description':'The department of the institution where the data was acquired.','units': 'n/a'},
                    'CapManufacturer': {'value':'n/a','description':'The manufacturer of the cap used in the study.','units': 'n/a'},
                    'CapManufacturersModelName': {'value':'n/a','description':'The model name of the cap used in the study.','units': 'n/a' },
                    'HeadCircumference': {'value':'n/a','description':'The head circumference of the subject.','units': units.centimeter},
                    'SubjectArtefactDescription': {'value':'n/a','description':'Description of any artefacts related to the subject.','units': 'n/a'    },
                    'TaskDescription': {'value':'n/a','description':'Description of the task performed by the subject.','units': 'n/a'},
                    'Instructions': {'value':'n/a','description':'Instructions given to the subject.','units': 'n/a'},
                    'CogPOID': {'value':'n/a','description':'The CogPO ID associated with the task.','units': 'n/a'},
                    'NIRSPlacementScheme': {'value':'n/a','description':'The NIRS placement scheme used.','units': 'n/a'},
                    'RecordingDuration': {'value': (data[intype].time[-1] - data[intype].time[0]).values, 'description': 'The duration of the recording.','units': units.seconds},
                    'SamplingFrequency': {'value': 1 / (data[intype].time[1] - data[intype].time[0]).values, 'description': 'The sampling frequency of the recording.','units': units.hertz},
                    'NIRSChannelCount': {'value': len(data[intype].channel), 'description': 'The number of NIRS channels.','units': 'n/a'},
                    'NIRSSourceOptodeCount': {'value': sum((data.geo2d.type == PointType(1).SOURCE).values), 'description': 'The number of NIRS source optodes.','units': 'n/a'},
                    'NIRSDetectorOptodeCount': {'value': sum((data.geo2d.type == PointType(1).DETECTOR).values), 'description': 'The number of NIRS detector optodes.','units': 'n/a'},
                    'ShortChannelCount': {'value': sum((xrutils.norm(data.geo3d.loc[data[intype].source] - data.geo3d.loc[data[intype].detector],  # noqa: E501
                                                           data.geo3d.points.crs) < units.millimeter * 15).values), 'description': 'The number of short channels.','units': 'n/a'},
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

    demofileTSV = Path(path) / 'participants.tsv'
    demofileTSV.parent.mkdir(parents=True, exist_ok=True)
    _strip_units_from_demographics(demographics).to_csv(demofileTSV, sep='\t', index=False)

    # Now save the JSON sidecar file with the column descriptions
    demofileJSON = Path(path) / 'participants.json'
    demofileJSON.parent.mkdir(parents=True, exist_ok=True)


    demo_info = {}

    for col in demographics.columns:
        description='n/a'
        if bids_descriptions is not None and col in bids_descriptions:
            description = bids_descriptions[col]

        if type(demographics[col][0]) == units.Quantity:
            demo_info[col] = {
                "Description": description,
                "Units": str(demographics[col][0].units)
            }
        else:
            demo_info[col] = {
                "Description": description,
                "Units": "n/a"
            }
      
    with open(demofileJSON, 'w') as f:
        json.dump(demo_info, f,indent=4)
    