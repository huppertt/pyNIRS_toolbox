"""Default tissue optical properties (port of nirs.media.tissues)."""
from pyBrainAnalyzIR.media.optical_properties import SpectralProp


def brain(wavelengths=(690.0, 830.0), so2=0.7, hbt=60.0):
    """Optical properties of brain tissue (port of nirs.media.tissues.brain)."""
    return SpectralProp(so2=so2, hbt=hbt, wavelengths=wavelengths)
