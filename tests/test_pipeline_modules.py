"""Tests for the pipeline modules and the option plumbing in ``pipeline.py``.

Every module in ``pyBrainAnalyzIR.pipelines.modules`` now stores its settings
as :class:`OptionVariable` objects inside an :class:`OptionsDict`.  These tests
check that contract for *all* modules generically, then exercise the pipeline
chaining/option-lookup behaviour used by ``examples/02_PipelineCreation.ipynb``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("cedalion")

from pyBrainAnalyzIR.dataclasses.options_variables import (  # noqa: E402
    OptionsDict,
    OptionVariable,
)
import pyBrainAnalyzIR.pipelines.modules.events as events        # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.filters as filters      # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.glm as glm              # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.mixedeffects as mixed   # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.motion_correction as mc  # noqa: E402
import pyBrainAnalyzIR.pipelines.modules.preproccessing as prep  # noqa: E402

pytestmark = pytest.mark.requires_cedalion


#: every concrete pipeline module shipped by the package
ALL_MODULES = [
    events.rename_stims,
    events.remove_stims,
    events.keep_stims,
    filters.bandpass_filter,
    filters.pca_filter,
    glm.GLM,
    mixed.MixedEffects,
    mc.motion_splineSG,
    mc.TDDR,
    mc.Wavelet,
    prep.resample,
    prep.intensity_opticaldensity,
    prep.opticaldensity_intensity,
    prep.conc2od,
    prep.mbll,
]

MODULE_IDS = [cls.__name__ for cls in ALL_MODULES]


# ---------------------------------------------------------------------------
# generic contract shared by every module
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_module_constructs_and_has_a_name(cls):
    job = cls()
    assert isinstance(job.name, str) and job.name


@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_module_options_is_an_optionsdict(cls):
    job = cls()
    assert isinstance(job.options, OptionsDict)


@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_every_option_is_an_option_variable(cls):
    """No module may still hold a bare python value in its options dict."""
    job = cls()
    for key, opt in job.options.options().items():
        assert isinstance(opt, OptionVariable), f"{cls.__name__}.{key} is not typed"


@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_every_option_documents_itself(cls):
    """Each option should carry a description and help text for the GUI."""
    job = cls()
    for key, opt in job.options.options().items():
        assert opt.description, f"{cls.__name__}.{key} has no description"
        assert opt.help, f"{cls.__name__}.{key} has no help text"


@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_options_start_at_their_defaults(cls):
    job = cls()
    for key, opt in job.options.options().items():
        assert opt.is_default, f"{cls.__name__}.{key} does not start at its default"


@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_reading_an_option_gives_a_plain_value(cls):
    """Module bodies use ``self.options['x']`` and need the raw value."""
    job = cls()
    for key in job.options:
        assert not isinstance(job.options[key], OptionVariable)


@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_reset_all_options_round_trips(cls):
    """Resetting an untouched module must leave every option at its default."""
    job = cls()
    job.reset_all_options()
    for key, opt in job.options.options().items():
        assert opt.is_default, f"{cls.__name__}.{key} is not at its default"


@pytest.mark.parametrize("cls", ALL_MODULES, ids=MODULE_IDS)
def test_module_help_and_show_do_not_raise(cls, capsys):
    job = cls()
    job.show()
    job.print_options()
    job.help()          # help() prints its output rather than returning it
    out = capsys.readouterr().out
    assert job.name in out


# ---------------------------------------------------------------------------
# specific option semantics
# ---------------------------------------------------------------------------

def test_glm_noise_model_is_restricted():
    job = glm.GLM()
    job.options["noise_model"] = "ar_irls"
    assert job.options["noise_model"] == "ar_irls"
    with pytest.raises(ValueError):
        job.options["noise_model"] = "ar-irls"     # typo used in the examples


def test_glm_ar_order_must_be_a_positive_integer():
    job = glm.GLM()
    job.options["ar_order"] = 16
    assert job.options["ar_order"] == 16
    with pytest.raises(ValueError):
        job.options["ar_order"] = 2.5


def test_resample_rate_must_be_positive():
    job = prep.resample()
    job.options["Fs"] = 4
    assert job.options["Fs"] == 4
    with pytest.raises(ValueError):
        job.options["Fs"] = 0


def test_splinesg_p_is_bounded():
    """``p`` is a smoothing fraction and must stay within [0, 1]."""
    job = mc.motion_splineSG()
    assert "p" in job.options
    job.options["p"] = 0.5
    with pytest.raises(ValueError):
        job.options["p"] = 1.5


def test_events_rename_takes_a_mapping():
    job = events.rename_stims()
    job.options["ListofChanges"] = {"1.0": "control", "2.0": "Tapping/Left"}
    assert job.options["ListofChanges"]["1.0"] == "control"
    with pytest.raises(ValueError):
        job.options["ListofChanges"] = ["not", "a", "mapping"]


def test_events_remove_takes_a_list():
    job = events.remove_stims()
    job.options["ListtoRemove"] = ["start marker"]
    assert job.options["ListtoRemove"] == ["start marker"]


def test_mixedeffects_formula_options():
    job = mixed.MixedEffects()
    job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:age"
    job.options["robust"] = True
    assert job.options["robust"] is True
    with pytest.raises(ValueError):
        job.options["robust"] = "yes"


# ---------------------------------------------------------------------------
# pipeline chaining and cross-module option lookup
# ---------------------------------------------------------------------------

def build_pipeline():
    """The pipeline used by the ROC example."""
    job = prep.intensity_opticaldensity()
    job = prep.mbll(job)
    job = prep.resample(job)
    job = glm.GLM(job)
    return job


def test_pipeline_chains_previous_jobs():
    job = build_pipeline()
    assert job.name == "GLM Model"
    assert job.previous_job is not None
    assert job.previous_job.name == "resample"


def test_get_option_searches_the_whole_pipeline():
    """``Fs`` lives on the resample step, not on the GLM tail."""
    job = build_pipeline()
    fs_option = job.get_option("Fs")
    assert isinstance(fs_option, OptionVariable)
    ar_option = job.get_option("ar_order")
    assert isinstance(ar_option, OptionVariable)


def test_get_option_raises_for_unknown_key():
    job = build_pipeline()
    with pytest.raises(KeyError):
        job.get_option("definitely_not_an_option")


def test_get_local_option_only_sees_this_module():
    job = build_pipeline()
    assert job.get_local_option("ar_order") is not None
    with pytest.raises(KeyError):
        job.get_local_option("Fs")


def test_set_all_options_applies_across_the_pipeline():
    job = build_pipeline()
    job.set_all_options({"noise_model": "ar_irls", "Fs": 4, "ar_order": 32})
    assert job.options["noise_model"] == "ar_irls"
    assert job.get_option("Fs").value == 4
    assert job.options["ar_order"] == 32


def test_set_all_options_rejects_invalid_values():
    job = build_pipeline()
    for bad in ({"noise_model": "ar-irls"}, {"Fs": 0}, {"ar_order": 2.5}):
        with pytest.raises(ValueError):
            job.set_all_options(bad)


def test_get_all_options_returns_values():
    job = build_pipeline()
    allopts = job.get_all_options()
    assert isinstance(allopts, dict)
    assert not any(isinstance(v, OptionVariable) for v in allopts.values())


def test_get_all_option_objects_returns_objects():
    job = build_pipeline()
    objects = job.get_all_option_objects()
    assert all(isinstance(v, OptionVariable) for v in objects.values())


def test_pipeline_reset_all_options():
    job = build_pipeline()
    job.set_all_options({"noise_model": "ar_irls", "Fs": 4})
    job.reset_all_options()
    assert job.options["noise_model"] == job.options.option("noise_model").default
    assert job.get_option("Fs").is_default


def test_pipeline_show_prints_each_step(capsys):
    job = build_pipeline()
    job.show()
    out = capsys.readouterr().out
    assert "GLM Model" in out
    assert "resample" in out
