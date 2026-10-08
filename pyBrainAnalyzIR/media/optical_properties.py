"""Optical properties of tissue (port of nirs.media.SpectralProp)."""
import numpy as np
import xarray as xr
import cedalion

from pyBrainAnalyzIR.media.spectra import getspectra

units = cedalion.units

SPEED_OF_LIGHT_MM_PER_S = 3e11


class SpectralProp:
    """Spectrally parameterized optical properties.

    Absorption is parameterized by oxygen saturation (so2), total hemoglobin
    (hbt, in uM), water and lipid fractions.  Reduced scattering follows the
    Mie power law ``musp = a * (lambda / 500) ** -b``.

    The dependent properties (mua, mus, musp, kappa, v) are returned as
    pint-quantified xr.DataArrays over the ``wavelength`` dimension, matching
    the cedalion conventions.
    """

    def __init__(self, so2=0.7, hbt=60.0, wavelengths=(690.0, 830.0), ri=1.45, g=0.89,
                 a=2.42, b=1.611, water=0.7, lipid=0.0):
        self.wavelengths = wavelengths
        self.so2 = so2
        self.hbt = hbt
        self.ri = ri
        self.g = g
        self.a = a  # mm^-1
        self.b = b
        self.water = water
        self.lipid = lipid

    # ---- validated attributes -------------------------------------------------
    @property
    def wavelengths(self):
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value):
        value = np.atleast_1d(np.asarray(value, dtype=float))
        if value.size == 0 or np.any(value <= 650) or np.any(value >= 900):
            raise ValueError("wavelengths must be within (650, 900) nm")
        self._wavelengths = value

    @property
    def so2(self):
        return self._so2

    @so2.setter
    def so2(self, value):
        if not 0 <= value <= 1:
            raise ValueError("so2 must be within [0, 1]")
        self._so2 = float(value)

    @property
    def hbt(self):
        return self._hbt

    @hbt.setter
    def hbt(self, value):
        if value < 0:
            raise ValueError("hbt must be non-negative")
        self._hbt = float(value)

    @property
    def ri(self):
        return self._ri

    @ri.setter
    def ri(self, value):
        if value < 1:
            raise ValueError("refractive index must be >= 1")
        self._ri = float(value)

    # ---- dependent properties ------------------------------------------------
    @property
    def hbo(self):
        return self.so2 * self.hbt

    @property
    def hbr(self):
        return (1 - self.so2) * self.hbt

    def _wl_array(self, values, unit):
        da = xr.DataArray(np.asarray(values, dtype=float), dims=["wavelength"],
                          coords={"wavelength": self.wavelengths})
        return da.pint.quantify(unit)

    @property
    def mua(self):
        """Absorption coefficient (1/mm)."""
        ext = getspectra(self.wavelengths)
        mua = (ext.sel(chromo="HbO") * self.hbo * 1e-6
               + ext.sel(chromo="HbR") * self.hbr * 1e-6
               + ext.sel(chromo="H2O") * self.water
               + ext.sel(chromo="Lipid") * self.lipid)
        return self._wl_array(mua.values, "1/mm")

    @property
    def musp(self):
        """Reduced scattering coefficient (1/mm)."""
        return self._wl_array(self.a * (self.wavelengths / 500) ** -self.b, "1/mm")

    @property
    def mus(self):
        """Scattering coefficient (1/mm)."""
        return self.musp / (1 - self.g)

    @property
    def v(self):
        """Speed of light in the medium (mm/s)."""
        return (SPEED_OF_LIGHT_MM_PER_S / self.ri) * units("mm/s")

    @property
    def kappa(self):
        """Diffusion coefficient 1/(3*mua + 3*musp) (mm)."""
        return 1 / (3 * self.mua + 3 * self.musp)

    def __repr__(self):
        return (f"SpectralProp(so2={self.so2}, hbt={self.hbt}, "
                f"wavelengths={self.wavelengths.tolist()}, water={self.water}, "
                f"lipid={self.lipid}, a={self.a}, b={self.b}, g={self.g}, ri={self.ri})")
