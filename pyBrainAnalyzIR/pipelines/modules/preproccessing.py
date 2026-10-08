import warnings

import numpy as np
import cedalion.nirs
import xarray as xr
import cedalion.math.resample
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, StringOption, ListOption, ObjectOption, BooleanOption)


class input_data(cedalion_module):
    # Module to handle input data
    # This is needed so you can specify the input timeseries name for the pipeline within the pipeline_manager GUI
    
    def __init__(self, previous_job=None):
        self.name = "Input Data"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'InputName': StringOption('amp', 'amp', description='Name of the input timeseries'),
                                    })
        self.inputName = self.options['InputName']
        self.outputName = self.options['InputName']
        self.description = "Handle input data for the pipeline"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        return rec

class resample(cedalion_module):
    # Module to resample the input fNIRS signal to a specified sampling frequency
    def __init__(self, previous_job=None):
        self.name = "Resample"
        self.advanced_module = False
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
        self.advanced_module = False
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
        self.name = "Calculate Raw Data from OD"
        self.advanced_module = True
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
        self.name = "Calculate OD from Concentration"
        self.advanced_module = True
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
        self.advanced_module = False
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


class TrimBaseline(cedalion_module):
    # Port of nirs.modules.TrimBaseline
    def __init__(self, previous_job=None):
        self.name = "Trim Pre/Post Baseline"
        self.advanced_module = False
        self._cite = None
        self.options = OptionsDict({
            'preBaseline': NumericOption(30, minimum=0, allow_none=True,
                                         description='Maximum baseline before the first stimulus (s)',
                                         help='Data more than this many seconds before the onset of the first '
                                              'stimulus event is removed. None keeps all of it.'),
            'postBaseline': NumericOption(30, minimum=0, allow_none=True,
                                          description='Maximum baseline after the last stimulus (s)',
                                          help='Data more than this many seconds after the end of the last '
                                               'stimulus event is removed. None keeps all of it.'),
            'resetTime': BooleanOption(False,
                                       description='Reset time to start at zero',
                                       help='If True, the time vector (and the stimulus onsets) are shifted so '
                                            'that the trimmed data starts at t=0.'),
            'Trim_Auxillary_Data': BooleanOption(True,
                                                 description='Also trim auxiliary data',
                                                 help='If True, the auxiliary time series (aux_ts) are trimmed '
                                                      '(and shifted) the same way as the data.'),
        })
        self.inputName = None
        self.outputName = None
        self.description = "Remove excessive baseline at the beginning or end of the recording"
        self.previous_job = previous_job

    @staticmethod
    def _trim(ts, t_min, t_max, tshift):
        if not (hasattr(ts, 'dims') and 'time' in ts.dims):
            return ts
        t = ts['time'].values
        ts = ts.isel(time=np.flatnonzero((t >= t_min) & (t <= t_max)))
        if tshift is not None:
            ts = ts.assign_coords(time=ts['time'] - tshift)
            if 'samples' in ts.coords:
                ts = ts.assign_coords(samples=('time', np.arange(ts.sizes['time'])))
        return ts

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec

        stim = rec.stim
        if stim is None or len(stim) == 0:
            warnings.warn('TrimBaseline called on file with no events')
            return rec

        pre = self.options['preBaseline']
        post = self.options['postBaseline']
        pre = np.inf if pre is None else pre
        post = np.inf if post is None else post
        onset = stim['onset'].to_numpy(dtype=float)
        t_min = onset.min() - pre
        t_max = (onset + stim['duration'].to_numpy(dtype=float)).max() + post

        containers = [rec.timeseries, rec.masks]
        if self.options['Trim_Auxillary_Data']:
            containers.append(rec.aux_ts)

        tshift = None
        if self.options['resetTime']:
            starts = [float(v['time'].values[(v['time'].values >= t_min) & (v['time'].values <= t_max)].min())
                      for c in containers for v in c.values()
                      if hasattr(v, 'dims') and 'time' in v.dims
                      and ((v['time'].values >= t_min) & (v['time'].values <= t_max)).any()]
            tshift = min(starts) if starts else None

        for container in containers:
            for key, value in list(container.items()):
                container[key] = self._trim(value, t_min, t_max, tshift)

        if tshift is not None:
            stim = stim.copy()
            stim['onset'] = stim['onset'] - tshift
            rec.stim = stim
        return rec
