"""Tabulated absorption spectra (port of nirs.media.getspectra).

HbO/HbR extinction coefficients come from cedalion's Prahl spectrum
(``cedalion.nirs.get_extinction_coefficients``), which is identical to the
table shipped with the MATLAB nirs-toolbox.  Water and lipid absorption
spectra are not provided by cedalion and are ported from the nirs-toolbox
``spectra.mat`` file.

References:
    Hemoglobin: S. Prahl, http://omlc.ogi.edu/spectra/hemoglobin/
    Fat: R.L.P. van Veen, H.J.C.M. Sterenborg, A. Pifferi, A. Torricelli and
        R. Cubeddu, http://omlc.ogi.edu/spectra/fat/
    Water: D. J. Segelstein, "The complex refractive index of water,"
        University of Missouri-Kansas City, (1981).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
import cedalion.nirs

_SPECTRA_FILE = Path(__file__).parent / "data" / "spectra.npz"


@lru_cache(maxsize=1)
def _load_tables():
    with np.load(_SPECTRA_FILE) as tables:
        return {key: tables[key] for key in tables.files}


def getspectra(wavelengths):
    """Return absorption spectra at the requested wavelengths.

    Args:
        wavelengths: array-like of wavelengths (nm).

    Returns:
        xr.DataArray with dims ("chromo", "wavelength") and chromo coordinates
        ["HbO", "HbR", "H2O", "Lipid"].  HbO/HbR are molar extinction
        coefficients in mm^-1/M (same convention as cedalion); H2O and Lipid
        are the absorption coefficients (mm^-1) of pure water / lipid, which
        are scaled by the volume fraction.  Because the rows have different
        units the array is not pint-quantified; units are listed in attrs.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    tables = _load_tables()

    hb = cedalion.nirs.get_extinction_coefficients("prahl", wavelengths)
    hb = hb.pint.dequantify().transpose("chromo", "wavelength").values

    water = np.interp(wavelengths, tables["water"][:, 0], tables["water"][:, 1])
    fat = np.interp(wavelengths, tables["fat"][:, 0], tables["fat"][:, 1])

    return xr.DataArray(
        np.vstack([hb, water, fat]),
        dims=["chromo", "wavelength"],
        coords={"chromo": ["HbO", "HbR", "H2O", "Lipid"], "wavelength": wavelengths},
        attrs={"units": {"HbO": "mm^-1 / M", "HbR": "mm^-1 / M",
                         "H2O": "mm^-1", "Lipid": "mm^-1"}},
    )
