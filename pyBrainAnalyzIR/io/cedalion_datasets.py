"""
    This file handles the import of datasets distributed as part of the cedalion library.
"""

import cedalion.dataclasses as cdc
from cedalion.datasets import DATASETS
import cedalion.io
import os.path  # noqa: F401
import pickle  # noqa: F401
from gzip import GzipFile  # noqa: F401
from pathlib import Path  # noqa: F401
import pandas as pd  # noqa: F401

import pooch
import xarray as xr  # noqa: F401
import pyBrainAnalyzIR.dataclasses.dataset as dataset

DatasetsValid = [
    'fingertapping.zip',
    'fingertappingDOT.zip',
    'multisubject-fingertapping.zip',
    'nn22_resting_state.zip'
]


def get_data(dataset_zip: str) -> cdc.Recording:
    """Retrieves a finger tapping DOT example dataset from the IBS Lab."""

    if not DATASETS.registry_files:
        raise ValueError("No datasets are registered in the cedalion library.")

    fnames = DATASETS.fetch(dataset_zip, processor=pooch.Unzip())

    dset = dataset.DataSet()

    for file in fnames:

        if file.endswith(".snirf"):
            rec = cedalion.io.read_snirf(file)[0]

            geo3d = rec.geo3d.points.rename({"NASION": "Nz"})
            geo3d = geo3d.rename({"pos": "digitized"})
            rec.geo3d = geo3d

            amp = rec.get_timeseries("amp")
            amp = amp.pint.dequantify().pint.quantify("V")
            rec.set_timeseries("amp", amp, overwrite=True)
            dset.import_data(rec)

    return dset
