import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import savemat

import cedalion.dataclasses as cdc

from pyBrainAnalyzIR.io import loadDirectory, loadDotNIRS, loadNIRx


def _dotnirs_payload():
    return {
        "d": np.array([[10, 20], [11, 21], [12, 22], [13, 23]], dtype=float),
        "t": np.array([0.0, 0.1, 0.2, 0.3]),
        "s": np.array([0, 1, 1, 0]),
        "aux": np.array([[1], [2], [3], [4]], dtype=float),
        "CondNames": np.array(["Tapping"], dtype=object),
        "SD": {
            "Lambda": np.array([760, 850]),
            "SrcPos": np.array([[0, 0, 0]], dtype=float),
            "DetPos": np.array([[30, 0, 0]], dtype=float),
            "MeasList": np.array([[1, 1, 1, 1], [1, 1, 1, 2]]),
            "auxChannels": np.array(["pulse"], dtype=object),
        },
        "demographics": {"age": np.array([[29]])},
    }


def _fake_snirf_recording(time):
    recording = cdc.Recording()
    recording["amp"] = xr.DataArray(
        np.ones((1, 2, len(time))),
        dims=("channel", "wavelength", "time"),
        coords={
            "channel": ["S1D1"],
            "source": ("channel", ["S1"]),
            "detector": ("channel", ["D1"]),
            "wavelength": [760, 850],
            "time": time,
        },
    )
    recording.stim = pd.DataFrame(
        columns=["onset", "duration", "value", "trial_type"]
    )
    return recording


def test_load_dot_nirs_maps_signal_probe_events_and_auxiliary(tmp_path):
    source = tmp_path / "pilot.nirs"
    savemat(source, _dotnirs_payload())

    dataset = loadDotNIRS(source)

    assert len(dataset.dataset) == 1
    recording = dataset.dataset[0]
    amp = recording.timeseries["amp"]
    assert amp.shape == (1, 2, 4)
    np.testing.assert_array_equal(amp.time.values, [0.0, 0.1, 0.2, 0.3])
    assert amp.channel.values.tolist() == ["S1D1"]
    assert amp.wavelength.values.tolist() == [760.0, 850.0]
    assert recording.geo2d.label.values.tolist() == ["S1", "D1"]
    assert recording.geo3d.label.values.tolist() == ["S1", "D1"]
    assert recording.stim.trial_type.tolist() == ["Tapping"]
    assert recording.stim.duration.tolist() == pytest.approx([0.2])
    assert recording.aux_ts["aux"].channel.values.tolist() == ["pulse"]
    assert recording.meta_data["age"] == 29


def test_load_dot_nirs_skips_nirx_sibling_unless_forced(tmp_path):
    source = tmp_path / "run.nirs"
    savemat(source, _dotnirs_payload())
    source.with_suffix(".wl1").touch()

    assert len(loadDotNIRS(source).dataset) == 0
    assert len(loadDotNIRS(source, force=True).dataset) == 1


def test_load_dot_nirs_accepts_a_single_measurement_row(tmp_path):
    source = tmp_path / "single-channel.nirs"
    payload = _dotnirs_payload()
    payload["d"] = payload["d"][:, :1]
    payload["SD"]["Lambda"] = np.array([760])
    payload["SD"]["MeasList"] = np.array([[1, 1, 1, 1]])
    savemat(source, payload)

    recording = loadDotNIRS(source).dataset[0]

    assert recording.timeseries["amp"].shape == (1, 1, 4)


def test_load_nirx_preserves_sidecars_and_merges_dotnirs_aux(tmp_path, monkeypatch):
    source = tmp_path / "run.snirf"
    source.touch()
    (tmp_path / "run_config.json").write_text('{"sampling_rate": 10}')
    savemat(tmp_path / "run.nirs", _dotnirs_payload())
    time = np.array([0.0, 0.1, 0.2, 0.30000001])
    monkeypatch.setattr(
        "cedalion.io.snirf.read_snirf",
        lambda path: [_fake_snirf_recording(time)] if path.exists() else [],
    )

    dataset = loadNIRx(source)

    assert len(dataset.dataset) == 1
    recording = dataset.dataset[0]
    assert recording.aux_obj["nirx_sidecars"]["metadata"]["run_config.json"] == {
        "sampling_rate": 10
    }
    assert "aux" in recording.aux_ts
    assert recording.stim.trial_type.tolist() == ["Tapping"]


def test_load_directory_deduplicates_snirf_and_matching_dotnirs(
    tmp_path, monkeypatch
):
    (tmp_path / "run.snirf").touch()
    savemat(tmp_path / "run.nirs", _dotnirs_payload())
    monkeypatch.setattr(
        "cedalion.io.snirf.read_snirf",
        lambda path: (
            [_fake_snirf_recording([0.0, 0.1, 0.2, 0.3])]
            if path.exists() else []
        ),
    )

    dataset = loadDirectory(tmp_path)

    assert len(dataset.dataset) == 1
    assert dataset.dataset[0].meta_data["NIRxSourceFile"].endswith("run.snirf")


def test_load_nirx_reports_unsupported_raw_only_folder(tmp_path):
    (tmp_path / "run.wl1").touch()

    with pytest.raises(NotImplementedError, match="raw .wl1/.wl2"):
        loadNIRx(tmp_path)
