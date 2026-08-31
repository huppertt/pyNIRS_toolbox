import cedalion.nirs
import xarray as xr
import cedalion.math.resample
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, StringOption, ListOption, ObjectOption)


class resample(cedalion_module):
    # Module to resample the input fNIRS signal to a specified sampling frequency
    def __init__(self, previous_job=None):
        self.name = "resample"
        self._cite = None
        self.options = OptionsDict({
            'Fs': NumericOption(4, minimum=0, inclusive=False,
                                description='Target sampling frequency (Hz)',
                                help='The data are resampled to this sampling rate. '
                                     'Must be strictly positive; values well below the '
                                     'original rate will low-pass the data.'),
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = "Resample the input fNIRS signal to a specified sampling frequency"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            if (self.inputName == 'last'):
                inputName = list(rec.timeseries.keys())[-1]
            else:
                inputName = self.inputName
            if (self.outputName == 'last'):
                outputName = list(rec.timeseries.keys())[-1]
            else:
                outputName = self.outputName

            rec[outputName] = cedalion.math.resample.resample(rec[inputName], self.options['Fs'])
            return rec


class intensity_opticaldensity(cedalion_module):
    # Module to convert raw intensity data to optical density
    def __init__(self, previous_job=None):
        self.name = "Calculate Optical Density"
        self._cite = None
        self.options = OptionsDict({})
        self.inputName = 'amp'
        self.outputName = 'od'
        self.description = "Convert raw intensity data to optical density"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            rec[self.outputName] = cedalion.nirs.int2od(rec[self.inputName], return_baseline=False)

            return rec


class opticaldensity_intensity(cedalion_module):
    # Module to convert optical density data back to raw intensity
    def __init__(self, previous_job=None):
        self.name = "Calculate raw data from OD"
        self._cite = None
        self.options = OptionsDict({
            'baseline': ObjectOption(None, allow_none=True,
                                     description='Baseline intensity',
                                     help='Baseline intensity used to invert the optical '
                                          'density. If None, a flat baseline of 100 is '
                                          'used for every channel.'),
        })
        self.inputName = 'od'
        self.outputName = 'amp'
        self.description = "Convert optical density data back to raw intensity"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            if (self.options['baseline'] is None):
                baseline = rec[self.inputName].mean("time")
                baseline[:] = 100
            else:
                baseline = self.options['baseline']

            rec[self.outputName] = cedalion.nirs.od2int(rec[self.inputName], baseline)
            return rec


class conc2od(cedalion_module):
    # Module to convert concentration data to optical density
    def __init__(self, previous_job=None):
        self.name = "Calculate OD from concentration"
        self._cite = "Cope & Delpy"
        self.options = OptionsDict({
            'spectrum': StringOption("prahl", allowed=['prahl'],
                                     description='Extinction coefficient spectrum',
                                     help='Name of the tabulated extinction coefficient '
                                          'spectrum used to convert between '
                                          'concentration and optical density.'),
            'dpf': ListOption([6, 6],
                              item_option=NumericOption(6, minimum=0, inclusive=False),
                              min_length=2, max_length=2,
                              description='Differential pathlength factors',
                              help='Differential pathlength factor for each wavelength, '
                                   'in the same order as the wavelengths of the probe.'),
        })
        self.inputName = 'conc'
        self.outputName = 'od'
        self.description = "Convert concentration data to optical density"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            dpf = xr.DataArray(self.options['dpf'], dims="wavelength",
                               coords={"wavelength": rec["amp"].wavelength})

            rec[self.outputName] = cedalion.nirs.conc2od(rec[self.inputName],
                                                         rec.geo3d, dpf, self.options['spectrum'])
            return rec


class mbll(cedalion_module):
    # Module to calculate concentration changes using the Modified Beer-Lambert Law
    def __init__(self, previous_job=None):
        self.name = "Calculate Modified Beer-Lambert"
        self._cite = "Cope & Delpy"
        self.options = OptionsDict({
            'spectrum': StringOption("prahl", allowed=['prahl'],
                                     description='Extinction coefficient spectrum',
                                     help='Name of the tabulated extinction coefficient '
                                          'spectrum used to convert between '
                                          'concentration and optical density.'),
            'dpf': ListOption([6, 6],
                              item_option=NumericOption(6, minimum=0, inclusive=False),
                              min_length=2, max_length=2,
                              description='Differential pathlength factors',
                              help='Differential pathlength factor for each wavelength, '
                                   'in the same order as the wavelengths of the probe.'),
        })
        self.inputName = 'od'
        self.outputName = 'conc'
        self.description = "Calculate concentration changes using the Modified Beer-Lambert Law"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            dpf = xr.DataArray(self.options['dpf'], dims="wavelength",
                               coords={"wavelength": rec[self.inputName].wavelength})

            rec[self.outputName] = cedalion.nirs.od2conc(rec[self.inputName],
                                                         rec.geo3d, dpf, self.options['spectrum'])
            return rec
