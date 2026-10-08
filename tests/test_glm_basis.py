"""Tests for the predefined GLM basis sets (``glm.BasisOptions``)."""
from __future__ import annotations

import copy
import importlib
import json

import numpy as np
import pytest

pytest.importorskip("cedalion")

import cedalion                                                    # noqa: E402
from cedalion.models.glm import basis_functions as bf              # noqa: E402

import pyBrainAnalyzIR.pipelines.modules.glm as glm                # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep    # noqa: E402
from pyBrainAnalyzIR.dataclasses.options_variables import PresetOption  # noqa: E402

units = cedalion.units
pytestmark = pytest.mark.requires_cedalion


# ---------------------------------------------------------------------------
# option semantics
# ---------------------------------------------------------------------------

def test_default_basis_is_the_gamma_preset():
    job = glm.GLM()
    opt = job.options.option("basis_function")
    assert isinstance(opt, PresetOption)
    assert opt.preset_name == "gamma" and opt.is_default
    basis = job.options["basis_function"]
    assert isinstance(basis, bf.Gamma)
    assert basis.sigma == 3 * units.s and basis.T == 3 * units.s


@pytest.mark.parametrize("key,cls", [
    ("gamma", bf.Gamma),
    ("gamma_deriv", bf.GammaDeriv),
    ("AFNIGamma", bf.AFNIGamma),
    ("GaussianKernels", bf.GaussianKernels),
    ("GaussianKernelsWithTails", bf.GaussianKernelsWithTails),
    ("boxcar", bf.DiracDelta),
])
def test_every_preset_resolves_to_its_basis_class(key, cls):
    job = glm.GLM()
    job.options["basis_function"] = key
    assert type(job.options["basis_function"]) is cls
    assert job.options.option("basis_function").preset_name == key


def test_preset_names_are_matched_loosely():
    job = glm.GLM()
    job.options["basis_function"] = "afni gamma"
    assert job.options.option("basis_function").preset_name == "AFNIGamma"
    job.options["basis_function"] = "GAMMA-DERIV"
    assert job.options.option("basis_function").preset_name == "gamma_deriv"


def test_resolved_preset_is_a_copy():
    job = glm.GLM()
    basis = job.options["basis_function"]
    basis.sigma = 99 * units.s
    assert glm.BasisOptions["gamma"].sigma == 3 * units.s
    assert job.options["basis_function"].sigma == 3 * units.s


def test_custom_basis_object_is_accepted():
    job = glm.GLM()
    custom = bf.Gamma(tau=1 * units.s, sigma=2 * units.s, T=0 * units.s)
    job.options["basis_function"] = custom
    opt = job.options.option("basis_function")
    assert job.options["basis_function"] is custom
    assert opt.preset_name is None and not opt.is_default
    assert str(opt).startswith("custom: Gamma(")


@pytest.mark.parametrize("bad", ["not_a_basis", 5, None])
def test_invalid_basis_is_rejected(bad):
    job = glm.GLM()
    with pytest.raises(ValueError):
        job.options["basis_function"] = bad
    assert job.options.option("basis_function").preset_name == "gamma"


def test_reset_restores_the_default_preset():
    job = glm.GLM()
    job.options["basis_function"] = "boxcar"
    job.options.option("basis_function").reset()
    assert job.options.option("basis_function").preset_name == "gamma"


def test_help_lists_the_presets():
    text = glm.GLM().options.option("basis_function").format_help()
    for key in glm.BasisOptions:
        assert key in text


def test_get_basis_function_helper():
    assert isinstance(glm.get_basis_function("AFNIGamma"), bf.AFNIGamma)
    custom = bf.DiracDelta()
    assert glm.get_basis_function(custom) is custom
    with pytest.raises(TypeError):
        glm.get_basis_function(3)


def test_options_deepcopy_keeps_the_selection():
    job = glm.GLM()
    job.options["basis_function"] = "AFNIGamma"
    clone = copy.deepcopy(job.options)
    assert clone.option("basis_function").preset_name == "AFNIGamma"


# ---------------------------------------------------------------------------
# running the GLM
# ---------------------------------------------------------------------------

def _run_glm(rec, basis):
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = prep.resample(job)
    job = glm.GLM(job)
    job.set_all_options({"Fs": 2, "noise_model": "ols", "verbose": False})
    job.options["basis_function"] = basis
    return job.run(rec)["stats"]


@pytest.mark.parametrize("basis", [
    "AFNIGamma",
    "GaussianKernels",
    bf.Gamma(tau=0 * units.s, sigma=2 * units.s, T=5 * units.s),
])
def test_glm_runs_with_preset_and_custom_basis(simulated_recording, basis):
    rec, _ = simulated_recording
    stats = _run_glm(rec, basis)
    betas = np.asarray(stats.betas)
    assert betas.size > 0 and np.isfinite(betas).all()


# ---------------------------------------------------------------------------
# pipeline manager GUI
# ---------------------------------------------------------------------------

@pytest.fixture()
def pm():
    pytest.importorskip("PySide6.QtWidgets")
    return importlib.import_module("pyBrainAnalyzIR.vis.pipeline_manager")


def test_gui_offers_the_presets_in_a_combo(pm):
    opt = glm.GLM().options.option("basis_function")
    assert [label for label, _ in pm._combo_choices(opt)] == list(glm.BasisOptions)


def test_gui_keeps_a_custom_object_selectable(pm):
    job = glm.GLM()
    custom = bf.DiracDelta()
    job.options["basis_function"] = custom
    choices = pm._combo_choices(job.options.option("basis_function"))
    assert choices[0][1] is custom
    assert [label for label, _ in choices[1:]] == list(glm.BasisOptions)


def test_json_round_trip_keeps_the_preset(pm):
    job = glm.GLM()
    job.options["basis_function"] = "GaussianKernels"
    payload = pm.pipeline_to_json([job])
    stored = json.loads(payload)["pipeline"][0]["options"]["basis_function"]
    assert stored == "GaussianKernels"
    restored = pm.pipeline_from_json(payload)[0]
    assert restored.options.option("basis_function").preset_name == "GaussianKernels"
    assert isinstance(restored.options["basis_function"], bf.GaussianKernels)


def test_json_with_custom_basis_falls_back_to_default(pm):
    job = glm.GLM()
    job.options["basis_function"] = bf.DiracDelta()
    restored = pm.pipeline_from_json(pm.pipeline_to_json([job]))[0]
    assert restored.options.option("basis_function").preset_name == "gamma"
