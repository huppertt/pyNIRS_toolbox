from pathlib import Path
import pandas as pd
import json
from cedalion.dataclasses.geometry import PointType
import cedalion.xrutils as xrutils


from cedalion import units
import pyBrainAnalyzIR
import pyBrainAnalyzIR.dataclasses.dataset as dataset


"""# BIDS Dataset
This dataset is organized according to the Brain Imaging Data Structure (BIDS) standard. For more information about BIDS, please visit [https://bids.neuroimaging.io/](https://bids.neuroimaging.io/).  # noqa: E501
"""


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
    bids_README_template['Experimental design (optional)'] = (
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
    bids_README_template['Stimuli (optional)'] = (
        "What was presented, and where stimulus files live (e.g. stimuli/)."
    )
    bids_README_template['Additional data acquired (optional)'] = (
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
    bids_README_template['How to use these data (optional)'] = (
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
    bids_dataset_description_template['DatasetDOI'] = "doi"

    return bids_dataset_description_template


def write_bids_readme(path: str | Path, description: dict):
    """Write the dataset_description.json file in the BIDS dataset directory.

    Args:
        path (str | Path): The directory path where the dataset_description.json file will be saved.
        description (dict): The content to be written in the dataset_description.json file.
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

    added_keys = [key for key in description.keys() if key not in key_order]
    key_order.extend(added_keys)

    readme_path = Path(path) / 'README.md'
    readme_path.parent.mkdir(parents=True, exist_ok=True)

    with open(readme_path, 'w') as f:
        for key in key_order:
            if key in description:
                f.write(f"### {key}\n")
                f.write(f"{description[key]}\n\n")


def write_bids_dataset_description(path: str | Path, content: dict):
    """Write a dataset_description.json file in the BIDS dataset directory.

    Args:
        path (str | Path): The directory path where the dataset_description.json file will be saved.
        content (dict): The content to be written in the dataset_description.json file.
    """
    description_path = Path(path) / 'dataset_description.json'
    description_path.parent.mkdir(parents=True, exist_ok=True)

    with open(description_path, 'w') as f:
        json.dump(content, f, indent=4)


def save_dataset_to_bids(dset: dataset.DataSet, path: str | Path):
    """Save the dataset in BIDS format.

    Args:
        dset (dataset.DataSet): The dataset to be saved.
        path (str | Path): The directory path where the BIDS dataset will be saved.
    """
    add_missing_bids_to_metadata(dset)
    demographics = dset.get_demographics()

    save_bids_demographics(demographics, demofile=Path(path) / 'participants.tsv')


def add_missing_bids_to_metadata(data, intype='amp'):
    """Add missing BIDS metadata to the existing metadata dictionary.

    Args:
        data: The data object containing metadata and other relevant information.

    """
    if (data.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
        for r in data.dataset:
            add_missing_bids_to_metadata(r, intype=intype)
        return
    else:
        defaults = {'Manufacturer': 'n/a',
                    'ManufacturersModelName': 'n/a',
                    'SoftwareVersions': 'n/a',
                    'DeviceSerialNumber': 'n/a',
                    'HardwareFilters': 'n/a',
                    'SourceType': 'n/a',
                    'DetectorType': 'n/a',
                    'InstitutionName': 'n/a',
                    'InstitutionAddress': 'n/a',
                    'InstitutionalDepartmentName': 'n/a',
                    'CapManufacturer': 'n/a',
                    'CapManufacturersModelName': 'n/a',
                    'HeadCircumference': 'n/a',
                    'SubjectArtefactDescription': 'n/a',
                    'TaskDescription': 'n/a',
                    'Instructions': 'n/a',
                    'CogPOID': 'n/a',
                    'NIRSPlacementScheme': 'n/a',
                    'RecordingDuration': (data[intype].time[-1] - data[intype].time[0]).values,
                    'SamplingFrequency': 1 / (data[intype].time[1] - data[intype].time[0]).values,
                    'NIRSChannelCount': len(data[intype].channel),
                    'NIRSSourceOptodeCount': sum((data.geo2d.type == PointType(1).SOURCE).values),
                    'NIRSDetectorOptodeCount': sum((data.geo2d.type == PointType(1).DETECTOR).values),
                    'ShortChannelCount': sum((xrutils.norm(data.geo3d.loc[data[intype].source] - data.geo3d.loc[data[intype].detector],  # noqa: E501
                                                           data.geo3d.points.crs) < units.millimeter * 15).values),
                    }
        for key, value in defaults.items():
            if key not in data.meta_data:
                data.meta_data[key] = value

        return


def save_bids_demographics(demographics: pd.DataFrame, demofile: str | Path):
    """Save the demographics dataframe in BIDS format.

    Args:
        demographics (pd.DataFrame): The demographics dataframe to be saved.
        demofile (str | Path): The file path where the demographics dataframe will be saved.
    """
    if not isinstance(demographics, pd.DataFrame):
        raise ValueError("demographics must be a pandas DataFrame")

    if not isinstance(demofile, (str, Path)):
        raise ValueError("demofile must be a string or Path")

    demofile = Path(demofile)
    demofile.parent.mkdir(parents=True, exist_ok=True)

    demographics.to_csv(demofile, sep='\t', index=False)
