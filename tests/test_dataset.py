"""Tests for the dataset metadata helpers."""
from __future__ import annotations

import json
from collections import OrderedDict

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
    write_bids_dataset,
)
from pyBrainAnalyzIR.testing import simData  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


class Recording:
    def __init__(self, meta_data):
        self.meta_data = OrderedDict(meta_data)


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
    assert saved["subject"].tolist() == ["subj_1", "subj_2"]
    assert demographics["age"][0].units == units.years

    with open(tmp_path / "participants.json") as f:
        sidecar = json.load(f)
    assert sidecar["age"]["Units"] == "year"


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

    with open(tmp_path / "participants.json") as f:
        participants_json = json.load(f)
    participants_json["age"]["Description"] = "Age from participants sidecar."
    with open(tmp_path / "participants.json", "w") as f:
        json.dump(participants_json, f, indent=4)

    root_fnirs_json = tmp_path / "task-finger_fnirs.json"
    with open(root_fnirs_json, "w") as f:
        json.dump({"HierarchyField": "root", "filedescription": "root"}, f)

    local_fnirs_json = next(path for path in tmp_path.rglob("*_fnirs.json") if path.parent != tmp_path)
    with open(local_fnirs_json) as f:
        local_metadata = json.load(f)
    local_metadata["filedescription"] = "local"
    with open(local_fnirs_json, "w") as f:
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

    def noop_add_missing_bids_to_metadata(data):
        assert data is dset
        return None

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

    write_bids_dataset(dset, tmp_path)
    snirf_file = next(tmp_path.rglob("*_fnirs.snirf"))
    fnirs_json = next(tmp_path.rglob("*_fnirs.json"))
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
    with open(fnirs_json) as f:
        assert json.load(f)["filedescription"] == "updated"
    saved_events = pd.read_csv(events_tsv, sep="\t")
    assert saved_events["trial_type"].tolist() == ["updated_event"]

    write_bids_dataset(dset, tmp_path, overwrite_snirf=True)

    assert snirf_file.read_text() == "updated"


def test_bids_dataset_splits_and_reads_derivative_snirf_files(tmp_path):
    rec, _ = simData.Data(snr=10)
    rec.meta_data["subject"] = "subj_1"
    rec.meta_data["task"] = "finger"
    rec["od"], _ = cedalion.nirs.int2od(rec["amp"], return_baseline=True)
    dset = DataSet([rec])

    write_bids_dataset(dset, tmp_path)

    raw_snirf = next(path for path in tmp_path.rglob("*_fnirs.snirf"))
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
    raw_snirf = next(path for path in tmp_path.rglob("*_fnirs.snirf"))
    assert list(read_snirf(raw_snirf)[0].timeseries.keys()) == ["amp"]
