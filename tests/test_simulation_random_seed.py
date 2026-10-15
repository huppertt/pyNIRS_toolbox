"""Reproducibility tests for simulation random seeds."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cedalion")

import pyBrainAnalyzIR.testing.simData as simdata  # noqa: E402
import pyBrainAnalyzIR.testing.simEvents as simevents  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


def test_event_designs_are_reproducible():
    blocked_a = simevents.blocked_stim_design(
        ncond=2, randomize_order=True, random_seed=13
    )
    blocked_b = simevents.blocked_stim_design(
        ncond=2, randomize_order=True, random_seed=13
    )
    random_a = simevents.rand_stim_design(ncond=2, random_seed=13)
    random_b = simevents.rand_stim_design(ncond=2, random_seed=13)

    assert blocked_a.equals(blocked_b)
    assert random_a.equals(random_b)


def test_ar_noise_and_data_are_reproducible():
    time = np.arange(1, 301) * 0.1
    noise_a = simdata.ARnoise(t=time, random_seed=19)
    noise_b = simdata.ARnoise(t=time, random_seed=19)
    np.testing.assert_array_equal(noise_a["amp"].values, noise_b["amp"].values)

    stim = simevents.rand_stim_design(t=time, ncond=1, random_seed=2)
    data_a, truth_a = simdata.Data(
        noise=simdata.ARnoise(t=time, random_seed=19),
        stim=stim.copy(),
        modifiers=[simdata.simMotionArtifact],
        random_seed=23,
    )
    data_b, truth_b = simdata.Data(
        noise=simdata.ARnoise(t=time, random_seed=19),
        stim=stim.copy(),
        modifiers=[simdata.simMotionArtifact],
        random_seed=23,
    )
    np.testing.assert_array_equal(data_a["amp"].values, data_b["amp"].values)
    np.testing.assert_array_equal(truth_a.values, truth_b.values)


def test_connectivity_simulations_are_reproducible():
    noise = simdata.ARnoise(t=np.arange(1, 301) * 0.1, random_seed=31)
    rec_a, truth_a = simdata.connectivity(
        noise=noise, pmax=2, random_seed=37
    )
    rec_b, truth_b = simdata.connectivity(
        noise=noise, pmax=2, random_seed=37
    )
    np.testing.assert_array_equal(rec_a["amp"].values, rec_b["amp"].values)
    np.testing.assert_array_equal(truth_a.values, truth_b.values)

    shortsep_a, shortsep_truth_a = simdata.connectivity_shortsep(
        sigma=0, t=np.arange(300) * 0.1, random_seed=41
    )
    shortsep_b, shortsep_truth_b = simdata.connectivity_shortsep(
        sigma=0, t=np.arange(300) * 0.1, random_seed=41
    )
    np.testing.assert_array_equal(
        shortsep_a["amp"].values, shortsep_b["amp"].values
    )
    np.testing.assert_array_equal(
        shortsep_truth_a.values, shortsep_truth_b.values
    )


def test_simdataset_is_reproducible_with_distinct_recordings():
    dataset_a, _ = simdata.simdataset(num_files=2, random_seed=29)
    dataset_b, _ = simdata.simdataset(num_files=2, random_seed=29)

    assert len(dataset_a.dataset) == len(dataset_b.dataset) == 2
    for rec_a, rec_b in zip(dataset_a.dataset, dataset_b.dataset):
        np.testing.assert_array_equal(rec_a["amp"].values, rec_b["amp"].values)

    first, second = dataset_a.dataset
    assert not np.array_equal(first["amp"].values, second["amp"].values)
