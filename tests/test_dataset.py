"""Tests for the dataset metadata helpers."""
from __future__ import annotations

import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cedalion")

import cedalion  # noqa: E402
from cedalion.io.snirf import read_snirf  # noqa: E402
from pyBrainAnalyzIR.dataclasses.dataset import DataSet  # noqa: E402
import pyBrainAnalyzIR.io.bids as bids_module  # noqa: E402
from pyBrainAnalyzIR.io.bids import (  # noqa: E402
    _get_participant_id,
    _get_session_name,
    _get_task_name,
    _apply_bids_stim_metadata,
    add_missing_bids_to_metadata,
    read_bids_data,
    read_bids_dataset,
    save_bids_demographics,
    write_bids_stim,
    write_bids_dataset,
)
from pyBrainAnalyzIR.testing import simData  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


class Recording:
    def __init__(self, meta_data):
        self.meta_data = OrderedDict(meta_data)
        self.stim = None
        self.timeseries = {}

    def __getitem__(self, key):
        return self.timeseries[key]


def test_bids_metadata_name_helpers_match_keys_case_insensitively():
    rec = Recording({
        "Subject_ID": "subj_1",
        "TaskName": "finger_tapping",
        "SESSION_ID": "baseline",
    })

    assert _get_participant_id(rec) == "subj_1"
    assert _get_task_name(rec) == "finger_tapping"
    assert _get_session_name(rec) == "baseline"


def test_bids_metadata_name_helpers_keep_defaults_when_keys_are_missing():
    rec = Recording({"unrelated": "value"})

    assert _get_participant_id(rec) is None
    assert _get_task_name(rec) == "func"
    assert _get_session_name(rec) is None


def test_add_demographics_units_updates_numeric_metadata():
    units = cedalion.units
    dset = DataSet([
        Recording({"age": 21.0, "group": "control", "enrolled": True}),
        Recording({"age": 34, "group": "patient", "enrolled": False}),
    ])

    dset.add_demographics_units("age", units.years)

    assert dset.dataset[0].meta_data["age"].magnitude == 21.0
    assert dset.dataset[0].meta_data["age"].units == units.years
    assert dset.dataset[1].meta_data["age"].magnitude == 34
    assert dset.dataset[1].meta_data["age"].units == units.years
    assert dset.dataset[0].meta_data["group"] == "control"
    assert dset.dataset[0].meta_data["enrolled"] is True

    dset.add_demographics_units("age", units.seconds)

    assert dset.dataset[0].meta_data["age"].magnitude == 21.0
    assert dset.dataset[0].meta_data["age"].units == units.seconds

    dset.add_demographics_units("age", None)

    assert dset.dataset[0].meta_data["age"] == 21.0
    assert dset.dataset[1].meta_data["age"] == 34


def test_add_demographics_units_warns_for_missing_metadata_key():
    dset = DataSet([Recording({"age": 21.0})])

    with pytest.warns(UserWarning, match="Metadata key 'missing'"):
        dset.add_demographics_units("missing", cedalion.units.years)


def test_get_stimulus_info_summarizes_events_across_scans():
    first = Recording({})
    first.stim = pd.DataFrame({
        "trial_type": ["left", "left", "right"],
        "duration": [1.0, 3.0, 2.0],
        "value": [2.0, 4.0, 6.0],
    })
    second = Recording({})
    second.stim = pd.DataFrame({
        "trial_type": ["left", "right", "right"],
        "duration": [5.0, 4.0, 6.0],
        "amplitude": [8.0, 10.0, 12.0],
    })
    third = Recording({})

    result = DataSet([first, second, third]).get_stimulus_info()

    assert result["task_types"] == ["left", "right"]
    event_info = result["event_info"].set_index("trial_type")
    assert event_info.loc["left"].to_dict() == {
        "total count": 3.0,
        "average_count": 1.0,
        "average_duration": 3.0,
        "average_amp": 14.0 / 3.0,
    }
    assert event_info.loc["right"].to_dict() == {
        "total count": 3.0,
        "average_count": 1.0,
        "average_duration": 4.0,
        "average_amp": 28.0 / 3.0,
    }

    assert len(result["scan_info"]) == 3
    first_info = result["scan_info"][0].set_index("trial_type")
    assert first_info.loc["left"].to_dict() == {
        "average_count": 2.0,
        "average_duration": 2.0,
        "average_amp": 3.0,
    }
    assert first_info.loc["right", "average_count"] == 1

    second_info = result["scan_info"][1].set_index("trial_type")
    assert second_info.loc["left", "average_count"] == 1
    assert second_info.loc["left", "average_amp"] == 8.0
    assert second_info.loc["right", "average_count"] == 2
    assert second_info.loc["right", "average_amp"] == 11.0

    third_info = result["scan_info"][2]
    assert third_info["average_count"].tolist() == [0, 0]
    assert third_info["average_duration"].isna().all()
    assert third_info["average_amp"].isna().all()


def test_get_stimulus_info_includes_meta_event_descriptions():
    first = Recording({})
    first.stim = pd.DataFrame({
        "trial_type": ["A", "B"],
        "duration": [1.0, 2.0],
        "value": [3.0, 4.0],
    })
    second = Recording({})
    second.stim = pd.DataFrame({
        "trial_type": ["A"],
        "duration": [3.0],
        "value": [5.0],
    })
    dset = DataSet([first, second])
    dset.add_meta_events_description("A", "A cognitive task.")

    result = dset.get_stimulus_info()

    event_info = result["event_info"].set_index("trial_type")
    assert event_info.loc["A", "description"] == "A cognitive task."
    assert pd.isna(event_info.loc["B", "description"])
    first_info = result["scan_info"][0].set_index("trial_type")
    second_info = result["scan_info"][1].set_index("trial_type")
    assert first_info.loc["A", "description"] == "A cognitive task."
    assert second_info.loc["A", "description"] == "A cognitive task."
    assert pd.isna(first_info.loc["B", "description"])


def test_get_stimulus_info_returns_empty_tables_without_stimuli():
    result = DataSet([Recording({})]).get_stimulus_info()

    assert result["task_types"] == []
    assert result["event_info"].columns.tolist() == [
        "trial_type",
        "total count",
        "average_count",
        "average_duration",
        "average_amp",
    ]
    assert len(result["scan_info"]) == 1
    assert result["scan_info"][0].columns.tolist() == [
        "trial_type",
        "average_count",
        "average_duration",
        "average_amp",
    ]
    assert result["scan_info"][0].empty


def test_add_meta_events_description_is_preserved_when_dataset_syncs():
    dset = DataSet([Recording({"subject": "A"}), Recording({"subject": "B"})])

    dset.add_meta_events_description("left", "The subject was tapping their left hand.")

    assert dset.dataset[0].meta_data["_bids_descriptions"]["event:left"] == (
        "The subject was tapping their left hand."
    )
    assert dset.dataset[1].meta_data["_bids_descriptions"]["event:left"] == (
        "The subject was tapping their left hand."
    )

    dset.add_demographics_by_index(pd.DataFrame({"age": [21, 22]}))
    assert dset.dataset[0].meta_data["_bids_descriptions"]["event:left"] == (
        "The subject was tapping their left hand."
    )


@pytest.mark.parametrize(
    ("eventkey", "description"),
    [("", "description"), ("left", "")],
)
def test_add_meta_events_description_rejects_empty_values(eventkey, description):
    dset = DataSet([Recording({})])

    with pytest.raises(ValueError):
        dset.add_meta_events_description(eventkey, description)


def test_dataset_bids_document_getters_merge_hidden_descriptors():
    rec = Recording({
        "subject": "A",
        "_bids_descriptors": {
            "readme": {"Overview": "Imported overview.", "Custom section": "Details."},
            "dataset_description": {
                "Name": "Imported dataset",
                "License": "CC-BY-4.0",
            },
        },
    })
    dset = DataSet([rec])

    readme = dset.readme()
    description = dset.data_description()

    assert readme["Overview"] == "Imported overview."
    assert readme["Custom section"] == "Details."
    assert "Methods" in readme
    assert description["Name"] == "Imported dataset"
    assert description["License"] == "CC-BY-4.0"
    assert "GeneratedBy" in description
    assert "_bids_descriptors" not in dset.get_demographics().columns


def test_add_meta_bids_documents_merges_values_across_recordings():
    dset = DataSet([
        Recording({"subject": "A"}),
        Recording({"subject": "B"}),
    ])
    readme = {"Overview": "Study overview.", "Custom section": "Extra details."}
    description = {"Name": "Study dataset", "License": "CC-BY-4.0"}

    dset.add_meta_readme(readme)
    dset.add_meta_dataset_description(description)

    for rec in dset.dataset:
        assert rec.meta_data["_bids_descriptors"] == {
            "readme": readme,
            "dataset_description": description,
        }
    assert dset.readme()["Custom section"] == "Extra details."
    assert dset.data_description()["Name"] == "Study dataset"

    readme["Overview"] = "Mutated input."
    description["Name"] = "Mutated input."
    assert dset.readme()["Overview"] == "Study overview."
    assert dset.data_description()["Name"] == "Study dataset"


def test_readme_include_flags_are_stored_and_default_to_true():
    dset = DataSet([
        Recording({"subject": "A"}),
        Recording({"subject": "B"}),
    ])

    default_flags = dset.readme_include()
    assert default_flags
    assert all(default_flags.values())

    dset.add_meta_readme_include({
        "Overview": False,
        "Custom section": False,
    })

    for rec in dset.dataset:
        assert rec.meta_data["_bids_descriptions"]["readme_include:Overview"] is False
        assert rec.meta_data["_bids_descriptions"]["readme_include:Custom section"] is False
    assert dset.readme_include()["Overview"] is False
    assert dset.readme_include()["Methods"] is True
    assert dset.readme_include()["Custom section"] is False


@pytest.mark.parametrize("method_name", [
    "add_meta_readme",
    "add_meta_dataset_description",
])
def test_add_meta_bids_documents_requires_nonempty_dataset(method_name):
    dset = DataSet()

    with pytest.raises(ValueError, match="empty dataset"):
        getattr(dset, method_name)({})


@pytest.mark.parametrize("method_name", [
    "add_meta_readme",
    "add_meta_dataset_description",
])
def test_add_meta_bids_documents_requires_dictionary(method_name):
    dset = DataSet([Recording({})])

    with pytest.raises(TypeError, match="must be a dictionary"):
        getattr(dset, method_name)([])


def test_read_bids_readme_preserves_plain_markdown(tmp_path):
    readme_path = tmp_path / "README.md"
    readme_path.write_text("A dataset overview without section headings.\n")

    assert bids_module._read_bids_readme(readme_path) == {
        "Overview": "A dataset overview without section headings."
    }


def test_add_missing_bids_metadata_adds_units_to_single_recording():
    rec, _ = simData.Data(snr=10)

    add_missing_bids_to_metadata(rec)

    assert rec.meta_data["HeadCircumference"] == "n/a"
    assert rec.meta_data["RecordingDuration"].units == cedalion.units.seconds
    assert rec.meta_data["SamplingFrequency"].units == cedalion.units.hertz
    duration_magnitude = rec.meta_data["RecordingDuration"].magnitude

    add_missing_bids_to_metadata(rec)

    assert rec.meta_data["RecordingDuration"].units == cedalion.units.seconds
    assert rec.meta_data["SamplingFrequency"].units == cedalion.units.hertz
    assert rec.meta_data["RecordingDuration"].magnitude == duration_magnitude


def test_save_bids_demographics_strips_units_from_tsv(tmp_path):
    units = cedalion.units
    demographics = pd.DataFrame({
        "subject": ["subj_1", "subj_2"],
        "age": [21 * units.years, 34 * units.years],
    })

    save_bids_demographics(demographics, tmp_path)

    saved = pd.read_csv(tmp_path / "participants.tsv", sep="\t")
    assert saved["age"].tolist() == [21, 34]
    assert saved["participant_id"].tolist() == ["sub-subj+1", "sub-subj+2"]
    assert demographics["age"][0].units == units.years

    with open(tmp_path / "participants.json", encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar["age"]["Units"] == "year"


def test_save_bids_demographics_keeps_custom_text_and_omits_recording_fields(tmp_path):
    demographics = pd.DataFrame({
        "subject": ["A"],
        "gender": ["M"],
        "study_site": ["north"],
        "task": ["finger"],
        "session": ["baseline"],
    })

    save_bids_demographics(
        demographics,
        tmp_path,
        bids_descriptions={"gender": "Participant gender", "study_site": "Recruitment site"},
    )

    written = pd.read_csv(tmp_path / "participants.tsv", sep="\t")
    assert written.columns.tolist() == ["participant_id", "gender", "study_site"]
    assert written.loc[0, "gender"] == "M"
    with open(tmp_path / "participants.json", encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar["gender"] == {"Description": "Participant gender"}
    assert sidecar["study_site"] == {"Description": "Recruitment site"}


def test_write_bids_events_uses_required_column_order_and_units(tmp_path):
    events = pd.DataFrame({
        "name": ["task"],
        "onset": [2.5],
        "duration": [1.0],
        "amplitude": [3.0],
        "response_time": [0.4],
    })
    events_path = tmp_path / "sub-A_task-test_events.tsv"

    write_bids_stim(events, events_path)

    written = pd.read_csv(events_path, sep="\t")
    assert written.columns.tolist() == [
        "onset", "duration", "trial_type", "value", "response_time"
    ]
    assert written.loc[0, "trial_type"] == "task"
    assert written.loc[0, "value"] == 3.0
    with open(events_path.with_suffix(".json"), encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar["onset"]["Units"] == "s"
    assert sidecar["duration"]["Units"] == "s"
    assert sidecar["response_time"]["Description"]
    assert sidecar["Description"]
    assert sidecar["StimulusPresentation"] == {"SoftwareName": "unknown"}
    assert sidecar["trial_type"]["Description"] == (
        "Task condition or event type associated with the event."
    )
    assert "Levels" not in sidecar["trial_type"]


def test_write_bids_events_describes_trial_type_levels(tmp_path):
    events = pd.DataFrame({
        "onset": [0.0, 1.0],
        "duration": [0.5, 0.5],
        "trial_type": ["left", "right"],
    })
    event_descriptions = {
        "left": "The subject was tapping their left hand.",
        "right": "The subject was tapping their right hand.",
    }
    events_path = tmp_path / "sub-A_task-motor_events.tsv"

    write_bids_stim(
        events,
        events_path,
        event_descriptions=event_descriptions,
    )

    with open(events_path.with_suffix(".json"), encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar["trial_type"]["Levels"] == event_descriptions
    assert "Description" in sidecar
    assert "left: The subject was tapping their left hand." in sidecar["Description"]
    assert "right: The subject was tapping their right hand." in sidecar["Description"]


def test_write_bids_events_uses_supplied_stimulus_presentation(tmp_path):
    presentation = {"SoftwareName": "PsychoPy", "SoftwareVersion": "2025.1"}
    events_path = tmp_path / "sub-A_task-test_events.tsv"
    events = pd.DataFrame({"onset": [0.0], "duration": [1.0]})

    write_bids_stim(events, events_path, stimulus_presentation=presentation)

    with open(events_path.with_suffix(".json"), encoding="utf-8") as f:
        sidecar = json.load(f)
    assert sidecar["StimulusPresentation"] == presentation


def test_write_bids_events_requires_onset_and_duration(tmp_path):
    with pytest.raises(ValueError, match="missing required column"):
        write_bids_stim(pd.DataFrame({"trial_type": ["task"]}), tmp_path / "events.tsv")


def test_write_bids_dataset_writes_nirs_channel_and_optode_sidecars(tmp_path):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    rec.meta_data["age"] = 34
    rec.meta_data["_bids_descriptors"] = {
        "readme": {
            "Overview": "Metadata overview.",
            "Saved section": "Saved from recording metadata.",
        },
        "dataset_description": {
            "Name": "Metadata dataset name",
            "License": "CC-BY-4.0",
            "GeneratedBy": [{"Name": "Metadata generator"}],
            "StimulusPresentation": {"SoftwareName": "Metadata stimulus software"},
        },
    }
    rec.stim = pd.DataFrame({
        "trial_type": ["left"],
        "onset": [0.0],
        "duration": [1.0],
        "value": [1.0],
    })
    dset = DataSet([rec])
    dset.add_meta_data_description("age", "Age of the participant in years.")
    dset.add_meta_events_description(
        "left", "The subject was tapping their left hand."
    )
    dset.add_meta_readme({"Added section": "Added through the DataSet API."})
    dset.add_meta_dataset_description({
        "HowToAcknowledge": "Stored through the DataSet API."
    })
    dset.add_meta_readme_include({
        "Methods": False,
    })

    write_bids_dataset(
        dset,
        tmp_path,
        readme={"Overview": "Call overview."},
        dataset_description={"Name": "Call dataset name"},
    )

    readme_text = (tmp_path / "README.md").read_text()
    assert "Call overview." in readme_text
    assert "### Overview" in readme_text
    assert "### Methods" not in readme_text
    assert "Saved from recording metadata." in readme_text
    assert "Added through the DataSet API." in readme_text
    assert "### Experimental design" in readme_text
    assert rec.meta_data["_bids_descriptions"]["readme_include:Methods"] is False
    with open(tmp_path / "dataset_description.json", encoding="utf-8") as f:
        dataset_description = json.load(f)
    assert dataset_description["Name"] == "Call dataset name"
    assert dataset_description["License"] == "CC-BY-4.0"
    assert dataset_description["HowToAcknowledge"] == (
        "Stored through the DataSet API."
    )
    assert dataset_description["GeneratedBy"] == [{"Name": "Metadata generator"}]
    assert "StimulusPresentation" not in dataset_description
    events_json = next(tmp_path.rglob("*_events.json"))
    with open(events_json, encoding="utf-8") as f:
        event_sidecar = json.load(f)
    assert event_sidecar["StimulusPresentation"] == {
        "SoftwareName": "Metadata stimulus software"
    }
    assert event_sidecar["trial_type"]["Levels"] == {
        "left": "The subject was tapping their left hand."
    }
    assert "The subject was tapping their left hand." in event_sidecar["Description"]

    with open(tmp_path / "participants.json", encoding="utf-8") as f:
        participants_sidecar = json.load(f)
    assert participants_sidecar["age"]["Description"] == (
        "Age of the participant in years."
    )

    channels_path = next(tmp_path.rglob("*_channels.tsv"))
    channels = pd.read_csv(channels_path, sep="\t")
    assert channels.columns.tolist() == [
        "name",
        "type",
        "source",
        "detector",
        "wavelength_nominal",
        "units",
        "description",
    ]
    assert len(channels) == len(rec["amp"].channel) * len(rec["amp"].wavelength)
    assert channels.loc[0].to_dict() == {
        "name": str(rec["amp"].channel.values[0]),
        "type": "NIRSCWAMPLITUDE",
        "source": str(rec["amp"].source.values[0]),
        "detector": str(rec["amp"].detector.values[0]),
        "wavelength_nominal": float(rec["amp"].wavelength.values[0]),
        "units": "V",
        "description": "fNIRS Channel between Optodes Source index 1 to "
        "Detector Index 1 at wavelength 760nm",
    }
    with open(channels_path.with_suffix(".json"), encoding="utf-8") as f:
        assert json.load(f)["Description"]
    assert rec.meta_data["NIRSChannelCount"] == len(channels)

    optodes_path = next(tmp_path.rglob("*_optodes.tsv"))
    optodes = pd.read_csv(optodes_path, sep="\t")
    assert optodes.columns.tolist() == [
        "name", "type", "x", "y", "z", "description"
    ]
    assert set(optodes["type"]) == {"source", "detector"}
    assert len(optodes) == len(set(channels["source"]) | set(channels["detector"]))
    assert optodes.loc[0, "description"] == "fNIRS Source Optode index 1"
    with open(optodes_path.with_suffix(".json"), encoding="utf-8") as f:
        assert json.load(f)["Description"]

    coordsystem_path = next(tmp_path.rglob("*_coordsystem.json"))
    with open(coordsystem_path, encoding="utf-8") as f:
        coordsystem = json.load(f)
    assert coordsystem["NIRSCoordinateSystem"] == "Other"
    assert coordsystem["NIRSCoordinateUnits"] == "mm"

    loaded = bids_module.read_bids_dataset(tmp_path)
    assert loaded.readme()["Overview"] == "Call overview."
    assert loaded.readme()["Saved section"] == "Saved from recording metadata."
    assert loaded.data_description()["Name"] == "Call dataset name"
    assert loaded.data_description()["License"] == "CC-BY-4.0"
    assert loaded.data_description()["StimulusPresentation"] == {
        "SoftwareName": "Metadata stimulus software"
    }


def test_read_bids_dataset_applies_sidecar_overrides(tmp_path):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    rec.meta_data["age"] = 21 * cedalion.units.years
    rec.meta_data["gender"] = "M"
    dset = DataSet([rec])
    write_bids_dataset(dset, tmp_path)

    participants = pd.read_csv(tmp_path / "participants.tsv", sep="\t", keep_default_na=False)
    participants.loc[0, "age"] = 99
    participants.loc[0, "gender"] = "X"
    participants["study_site"] = "north"
    participants.to_csv(tmp_path / "participants.tsv", sep="\t", index=False)

    with open(tmp_path / "participants.json", encoding="utf-8") as f:
        participants_json = json.load(f)
    participants_json["age"]["Description"] = "Age from participants sidecar."
    with open(tmp_path / "participants.json", "w", encoding="utf-8") as f:
        json.dump(participants_json, f, indent=4)

    root_fnirs_json = tmp_path / "task-finger_nirs.json"
    with open(root_fnirs_json, "w", encoding="utf-8") as f:
        json.dump({"HierarchyField": "root", "filedescription": "root"}, f)

    local_fnirs_json = next(path for path in tmp_path.rglob("*_nirs.json") if path.parent != tmp_path)
    with open(local_fnirs_json, encoding="utf-8") as f:
        local_metadata = json.load(f)
    local_metadata["filedescription"] = "local"
    with open(local_fnirs_json, "w", encoding="utf-8") as f:
        json.dump(local_metadata, f)

    root_events = pd.DataFrame({
        "trial_type": ["root_event"],
        "onset": [0.0],
        "duration": [1.0],
        "value": [1.0],
    })
    root_events.to_csv(tmp_path / "task-finger_events.tsv", sep="\t", index=False)

    local_events = pd.DataFrame({
        "trial_type": ["local_event"],
        "onset": [2.0],
        "duration": [3.0],
        "value": [4.0],
    })
    local_events.to_csv(
        next(path for path in tmp_path.rglob("*_events.tsv") if path.parent != tmp_path),
        sep="\t",
        index=False,
    )

    loaded = read_bids_dataset(tmp_path)
    loaded_rec = loaded.dataset[0]

    assert loaded_rec.meta_data["age"].magnitude == 99
    assert loaded_rec.meta_data["age"].units == cedalion.units.years
    assert loaded_rec.meta_data["gender"] == "X"
    assert loaded_rec.meta_data["study_site"] == "north"
    assert loaded_rec.meta_data["_bids_descriptions"]["age"] == "Age from participants sidecar."
    assert loaded_rec.meta_data["HierarchyField"] == "root"
    assert loaded_rec.meta_data["filedescription"] == "local"
    assert loaded_rec.stim["trial_type"].tolist() == ["local_event"]
    assert loaded_rec.stim["value"].tolist() == [4.0]
    loaded_demographics = loaded.get_demographics()
    assert loaded_demographics.loc[0, "age"].magnitude == 99
    assert loaded_demographics.loc[0, "gender"] == "X"
    assert loaded_demographics.loc[0, "study_site"] == "north"


def test_read_legacy_bids_stim_tsv(tmp_path):
    rec = Recording({})
    snirf_file = tmp_path / "sub-A" / "sub-A_task-finger_run-1_fnirs.snirf"
    snirf_file.parent.mkdir(parents=True)
    pd.DataFrame({
        "name": ["legacy_event"],
        "onset": [2.0],
        "duration": [3.0],
        "amplitude": [4.0],
    }).to_csv(
        snirf_file.with_name("sub-A_task-finger_run-1_stim.tsv"),
        sep="\t",
        index=False,
    )

    _apply_bids_stim_metadata(
        rec,
        snirf_file,
        tmp_path,
        {"sub": "A", "task": "finger", "run": "1"},
    )

    assert rec.stim["trial_type"].tolist() == ["legacy_event"]
    assert rec.stim["value"].tolist() == [4.0]


def test_write_bids_dataset_preserves_existing_snirf_by_default(tmp_path, monkeypatch):
    def fake_write_snirf(path, data):
        path.write_text(data.meta_data["filedescription"])

    existing = Recording({"filedescription": "original"})
    existing.timeseries = OrderedDict({"amp": "raw"})

    def noop_add_missing_bids_to_metadata(data):
        assert data is dset
        return None

    def noop_write_bids_nirs_sidecars(*args):
        assert len(args) == 5

    rec = Recording({
        "subject": "subj_1",
        "task": "finger",
        "filedescription": "original",
    })
    rec.timeseries = OrderedDict({"amp": "raw"})
    rec.stim = pd.DataFrame({
        "trial_type": ["original_event"],
        "onset": [0.0],
        "duration": [1.0],
        "value": [1.0],
    })
    dset = DataSet([rec])

    monkeypatch.setattr(
        bids_module, "add_missing_bids_to_metadata", noop_add_missing_bids_to_metadata
    )
    monkeypatch.setattr(bids_module, "write_snirf", fake_write_snirf)
    monkeypatch.setattr(
        bids_module, "read_snirf", lambda path: [existing] if path else []
    )
    monkeypatch.setattr(
        bids_module, "_write_bids_nirs_sidecars", noop_write_bids_nirs_sidecars
    )

    write_bids_dataset(dset, tmp_path)
    snirf_file = next(tmp_path.rglob("*_nirs.snirf"))
    fnirs_json = next(tmp_path.rglob("*_nirs.json"))
    events_tsv = next(tmp_path.rglob("*_events.tsv"))
    assert snirf_file.read_text() == "original"

    rec.meta_data["filedescription"] = "updated"
    rec.stim = pd.DataFrame({
        "trial_type": ["updated_event"],
        "onset": [2.0],
        "duration": [3.0],
        "value": [4.0],
    })

    write_bids_dataset(dset, tmp_path)

    assert snirf_file.read_text() == "original"
    with open(fnirs_json, encoding="utf-8") as f:
        assert json.load(f)["filedescription"] == "updated"
    saved_events = pd.read_csv(events_tsv, sep="\t")
    assert saved_events["trial_type"].tolist() == ["updated_event"]

    write_bids_dataset(dset, tmp_path, overwrite_snirf=True)

    assert snirf_file.read_text() == "original"

    rec.timeseries["amp"] = "changed raw"
    with pytest.raises(ValueError, match="Raw amplitude time course differs"):
        write_bids_dataset(dset, tmp_path)
    saved_events = pd.read_csv(events_tsv, sep="\t")
    assert saved_events["trial_type"].tolist() == ["updated_event"]


def test_bids_rewrite_updates_demographics_and_events_without_changing_snirf(tmp_path):
    rec, _ = simData.Data(snr=10, random_seed=17)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    rec.meta_data["age"] = 21 * cedalion.units.years
    rec.meta_data["gender"] = "M"
    rec.meta_data["study_site"] = "north"
    rec.stim = pd.DataFrame({
        "trial_type": ["left"],
        "onset": [1.0],
        "duration": [2.0],
        "value": [1.0],
    })

    dset = DataSet([rec])
    dset.add_meta_data_description("age", "Participant age.")
    dset.add_meta_data_description("gender", "Participant gender.")
    dset.add_meta_data_description("study_site", "Participant recruitment site.")
    dset.add_meta_events_description("left", "Left-hand tapping.")
    write_bids_dataset(dset, tmp_path, include_derivatives=False)

    snirf_file = next(tmp_path.rglob("*_nirs.snirf"))
    original_snirf_bytes = snirf_file.read_bytes()

    dset.add_demographics_by_index(pd.DataFrame({
        "age": [34 * cedalion.units.years],
        "gender": ["F"],
        "study_site": ["south"],
    }))
    rec.stim = pd.DataFrame({
        "trial_type": ["right", "right"],
        "onset": [4.5, 9.0],
        "duration": [1.25, 0.75],
        "value": [2.0, 3.0],
    })
    dset.add_meta_events_description("right", "Right-hand tapping.")

    write_bids_dataset(dset, tmp_path, include_derivatives=False)

    assert snirf_file.read_bytes() == original_snirf_bytes

    participants_tsv = pd.read_csv(
        tmp_path / "participants.tsv", sep="\t", keep_default_na=False
    )
    assert participants_tsv.loc[0, "age"] == 34
    assert participants_tsv.loc[0, "gender"] == "F"
    assert participants_tsv.loc[0, "study_site"] == "south"
    with open(tmp_path / "participants.json", encoding="utf-8") as f:
        participants_json = json.load(f)
    assert participants_json["age"] == {
        "Description": "Participant age.",
        "Units": "year",
    }
    assert participants_json["gender"]["Description"] == "Participant gender."
    assert participants_json["study_site"]["Description"] == (
        "Participant recruitment site."
    )

    events_tsv = next(tmp_path.rglob("*_events.tsv"))
    events = pd.read_csv(events_tsv, sep="\t")
    assert events["trial_type"].tolist() == ["right", "right"]
    assert events["onset"].tolist() == [4.5, 9.0]
    assert events["duration"].tolist() == [1.25, 0.75]
    assert events["value"].tolist() == [2.0, 3.0]
    with open(events_tsv.with_suffix(".json"), encoding="utf-8") as f:
        events_json = json.load(f)
    assert events_json["trial_type"]["Levels"] == {
        "right": "Right-hand tapping."
    }

    imported = read_bids_dataset(tmp_path, include_derivatives=False)
    imported_rec = imported.dataset[0]
    assert imported_rec.meta_data["age"].magnitude == 34
    assert imported_rec.meta_data["age"].units == cedalion.units.years
    assert imported_rec.meta_data["gender"] == "F"
    assert imported_rec.meta_data["study_site"] == "south"
    assert imported_rec.stim["trial_type"].tolist() == ["right", "right"]
    assert imported_rec.stim["onset"].tolist() == [4.5, 9.0]
    assert imported_rec.stim["duration"].tolist() == [1.25, 0.75]
    assert imported_rec.stim["value"].tolist() == [2.0, 3.0]
    assert snirf_file.read_bytes() == original_snirf_bytes


def test_bids_sidecars_override_snirf_and_missing_sidecars_fall_back(tmp_path):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    rec.meta_data["filedescription"] = "embedded value"
    rec.meta_data["MeasurementDate"] = "2024-05-01"
    rec.stim = pd.DataFrame({
        "trial_type": ["embedded_event"],
        "onset": [0.0],
        "duration": [1.0],
        "value": [1.0],
    })
    dset = DataSet([rec])
    write_bids_dataset(dset, tmp_path, include_derivatives=False)

    snirf_file = next(tmp_path.rglob("*_nirs.snirf"))
    original_bytes = snirf_file.read_bytes()
    original_snirf_metadata = read_snirf(snirf_file)[0].meta_data.copy()
    original_channels = rec["amp"].channel.values.astype(str).copy()
    original_sources = rec["amp"].source.values.astype(str).copy()
    original_labels = rec.geo3d.label.values.astype(str).copy()
    original_positions = rec.geo3d.pint.magnitude.copy()

    source_to_rename = original_sources[0]
    updated_sources = [
        "S-new" if source == source_to_rename else source
        for source in original_sources
    ]
    rec["amp"] = rec["amp"].assign_coords(
        source=("channel", updated_sources)
    )
    updated_labels = [
        "S-new" if label == source_to_rename else label
        for label in original_labels
    ]
    updated_positions = original_positions.copy()
    updated_positions[original_labels == source_to_rename, 0] = 123.0
    rec.geo3d = rec.geo3d.copy(
        data=updated_positions * rec.geo3d.pint.units
    ).assign_coords(label=("label", updated_labels))
    rec.meta_data["filedescription"] = "sidecar value"
    rec.meta_data["MeasurementDate"] = "2025-01-01"
    rec.stim = pd.DataFrame({
        "trial_type": ["sidecar_event"],
        "onset": [2.0],
        "duration": [3.0],
        "value": [4.0],
    })
    write_bids_dataset(
        dset, tmp_path, overwrite_snirf=True, include_derivatives=False
    )
    assert snirf_file.read_bytes() == original_bytes

    channels_path = next(tmp_path.rglob("*_channels.tsv"))
    channels = pd.read_csv(channels_path, sep="\t", keep_default_na=False)
    assert "S-new" in channels["source"].values
    assert channels.loc[channels["source"] == "S-new", "description"].str.contains(
        "Source index 1"
    ).all()

    optodes_path = next(tmp_path.rglob("*_optodes.tsv"))
    optodes = pd.read_csv(optodes_path, sep="\t", keep_default_na=False)
    assert optodes.loc[optodes["name"] == "S-new", "x"].iloc[0] == 123.0

    events_path = next(tmp_path.rglob("*_events.tsv"))
    nirs_json_path = next(tmp_path.rglob("*_nirs.json"))
    with open(nirs_json_path, encoding="utf-8") as f:
        nirs_json = json.load(f)
    assert nirs_json["filedescription"] == "sidecar value"

    loaded = read_bids_dataset(tmp_path, include_derivatives=False).dataset[0]
    assert loaded.meta_data["filedescription"] == "sidecar value"
    assert loaded.meta_data["MeasurementDate"] == "2025-01-01"
    assert loaded.stim["trial_type"].tolist() == ["sidecar_event"]
    assert loaded["amp"].source.values.astype(str).tolist() == updated_sources
    assert loaded.geo3d.sel(label="S-new").pint.magnitude[0] == 123.0
    assert snirf_file.read_bytes() == original_bytes

    nirs_json_path.unlink()
    events_path.unlink()
    channels_path.unlink()
    optodes_path.unlink()
    next(tmp_path.rglob("*_coordsystem.json")).unlink()
    (tmp_path / "participants.tsv").unlink()
    (tmp_path / "participants.json").unlink()
    loaded_without_sidecars = read_bids_dataset(
        tmp_path, include_derivatives=False
    ).dataset[0]
    assert loaded_without_sidecars.meta_data.get("MeasurementDate") == (
        original_snirf_metadata.get("MeasurementDate")
    )
    assert loaded_without_sidecars.stim["trial_type"].tolist() == ["embedded_event"]
    assert loaded_without_sidecars["amp"].channel.values.astype(str).tolist() == (
        original_channels.tolist()
    )
    assert loaded_without_sidecars["amp"].source.values.astype(str).tolist() == (
        original_sources.tolist()
    )
    assert loaded_without_sidecars.geo3d.label.values.astype(str).tolist() == (
        original_labels.tolist()
    )
    assert np.array_equal(
        loaded_without_sidecars.geo3d.pint.magnitude,
        original_positions,
    )


def test_write_bids_dataset_migrates_legacy_nirs_and_events_filenames(tmp_path):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    dset = DataSet([rec])
    write_bids_dataset(dset, tmp_path)

    snirf_file = next(tmp_path.rglob("*_nirs.snirf"))
    nirs_json = next(tmp_path.rglob("*_nirs.json"))
    original_bytes = snirf_file.read_bytes()
    legacy_snirf = snirf_file.with_name(snirf_file.name.replace("_nirs.snirf", "_fnirs.snirf"))
    legacy_json = nirs_json.with_name(nirs_json.name.replace("_nirs.json", "_fnirs.json"))
    snirf_file.rename(legacy_snirf)
    nirs_json.rename(legacy_json)
    events_file = next(tmp_path.rglob("*_events.tsv"))
    legacy_stim = events_file.with_name(events_file.name.replace("_events.tsv", "_stim.tsv"))
    legacy_stim_json = legacy_stim.with_suffix(".json")
    legacy_stim.write_text("name\tonset\tduration\nold\t0\t1\n")
    legacy_stim_json.write_text("{}")

    write_bids_dataset(dset, tmp_path)

    assert snirf_file.exists()
    assert snirf_file.read_bytes() == original_bytes
    assert not legacy_snirf.exists()
    assert nirs_json.exists()
    assert not legacy_json.exists()
    assert not legacy_stim.exists()
    assert not legacy_stim_json.exists()


def test_bids_dataset_splits_and_reads_derivative_snirf_files(tmp_path):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    rec["od"], _ = cedalion.nirs.int2od(rec["amp"], return_baseline=True)
    dset = DataSet([rec])

    write_bids_dataset(dset, tmp_path)

    raw_snirf = next(path for path in tmp_path.rglob("*_nirs.snirf"))
    derivative_snirf = (
        tmp_path
        / "bids_derivatives"
        / raw_snirf.relative_to(tmp_path).with_name(f"{raw_snirf.stem}_derivatives.snirf")
    )
    assert derivative_snirf.exists()
    assert list(read_snirf(raw_snirf)[0].timeseries.keys()) == ["amp"]
    assert list(read_snirf(derivative_snirf)[0].timeseries.keys()) == ["od"]

    loaded_with_derivatives = read_bids_dataset(tmp_path)
    loaded_without_derivatives = read_bids_dataset(tmp_path, include_derivatives=False)
    single_without_derivatives = read_bids_data(
        raw_snirf, tmp_path, include_derivatives=False
    )

    assert list(loaded_with_derivatives.dataset[0].timeseries.keys()) == ["amp", "od"]
    assert list(loaded_without_derivatives.dataset[0].timeseries.keys()) == ["amp"]
    assert list(single_without_derivatives[0].timeseries.keys()) == ["amp"]


def test_write_bids_dataset_can_skip_derivative_snirf_files(tmp_path):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    rec["od"], _ = cedalion.nirs.int2od(rec["amp"], return_baseline=True)
    dset = DataSet([rec])

    write_bids_dataset(dset, tmp_path, include_derivatives=False)

    assert not (tmp_path / "bids_derivatives").exists()
    raw_snirf = next(path for path in tmp_path.rglob("*_nirs.snirf"))
    assert list(read_snirf(raw_snirf)[0].timeseries.keys()) == ["amp"]
