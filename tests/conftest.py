"""Shared pytest fixtures and helpers for the pyBrainAnalyzIR test-suite.

The scientific stack that pyBrainAnalyzIR builds on (``cedalion`` in
particular) is an optional, heavyweight dependency that is not always
available -- for example on a bare CI runner.  Tests that genuinely need it
are marked with ``@pytest.mark.requires_cedalion`` and are skipped
automatically when the import fails, so the pure-python parts of the package
(most importantly the option-variable data classes) can still be verified
everywhere.
"""
from __future__ import annotations

import copy
import os

import pytest

# Matplotlib and Qt must never try to open a window during the test run.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _module_available(name: str) -> bool:
    """Return True when *name* can be imported."""
    try:
        __import__(name)
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


HAS_CEDALION = _module_available("cedalion")
HAS_QT = _module_available("PySide6.QtWidgets")

requires_cedalion = pytest.mark.skipif(
    not HAS_CEDALION, reason="cedalion is not installed in this environment"
)
requires_qt = pytest.mark.skipif(
    not HAS_QT, reason="PySide6 is not installed in this environment"
)


def pytest_collection_modifyitems(config, items):
    """Auto-skip marked tests when their optional dependency is missing."""
    skip_cedalion = pytest.mark.skip(reason="cedalion is not installed")
    skip_qt = pytest.mark.skip(reason="PySide6 is not installed")
    for item in items:
        if "requires_cedalion" in item.keywords and not HAS_CEDALION:
            item.add_marker(skip_cedalion)
        if "requires_qt" in item.keywords and not HAS_QT:
            item.add_marker(skip_qt)


@pytest.fixture(scope="session")
def qt_app():
    """A single QApplication shared by every Qt test in the session."""
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _no_modal_dialogs(request, monkeypatch):
    """Never let a test block on a modal Qt dialog.

    ``QMessageBox`` calls are synchronous and would hang a headless run
    forever, so they are replaced by no-ops for every Qt test.  Individual
    tests can still assert on them by patching the symbol themselves.
    """
    if not HAS_QT:
        return
    from PySide6.QtWidgets import QMessageBox

    for name in ("warning", "information", "critical", "about"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.Ok),
                            raising=False)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes),
                        raising=False)


@pytest.fixture(scope="session")
def _simulated_recording_source():
    """A small simulated fNIRS recording plus its ground-truth channel mask.

    Mirrors ``examples/01_SimulatingData.ipynb``.  Session-scoped because the
    autoregressive simulation is comparatively expensive.
    """
    pytest.importorskip("cedalion")
    import pyBrainAnalyzIR.testing.simData as simdata

    import numpy as np

    np.random.seed(0)
    rec, truth = simdata.Data(snr=25)
    return rec, truth


@pytest.fixture()
def simulated_recording(_simulated_recording_source):
    """A fresh copy of the simulated recording.

    ``cedalion_module.run()`` mutates the recording it is given (it adds new
    timeseries such as ``od``/``conc`` and the ``stats`` result), so each test
    must start from its own copy to stay independent of execution order.
    """
    rec, truth = _simulated_recording_source
    return copy.deepcopy(rec), copy.deepcopy(truth)


@pytest.fixture(scope="session")
def _simulated_dataset_source():
    """A small multi-file DataSet with demographics, as used for group stats."""
    pytest.importorskip("cedalion")
    import numpy as np
    import pandas as pd

    import pyBrainAnalyzIR.dataclasses.dataset as dataset
    import pyBrainAnalyzIR.testing.simData as simdata

    np.random.seed(1)
    dset = dataset.DataSet()
    for _ in range(4):
        rec, _ = simdata.Data(snr=10)
        dset.import_data(rec)

    demo = pd.DataFrame({
        "subject": ["A", "B", "C", "D"],
        "gender": ["M", "F", "M", "F"],
        "age": [21.0, 34.0, 27.0, 45.0],
    })
    dset.add_demographics_by_index(demo)
    return dset


@pytest.fixture()
def simulated_dataset(_simulated_dataset_source):
    """A fresh copy of the multi-file dataset (see ``simulated_recording``)."""
    return copy.deepcopy(_simulated_dataset_source)
