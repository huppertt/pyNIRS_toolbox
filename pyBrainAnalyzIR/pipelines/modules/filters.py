import cedalion.nirs

import pyBrainAnalyzIR.pipelines

import cedalion.sigproc.frequency as freq
import pyBrainAnalyzIR.sigproc.pca_filter
from pyBrainAnalyzIR.sigproc.short_separation import (
    DEFAULT_THRESHOLD, short_separation_filter, remove_short_channels_recording)

import pyBrainAnalyzIR.pipelines.pipeline
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, BooleanOption, QuantityOption)

units = cedalion.units


class bandpass_filter(cedalion_module):
    # Module to apply a band-pass filter to the input signal
    def __init__(self, previous_job=None):
        self.name = "Band-Pass Filter"
        self.advanced_module = False
        self._cite = None
        self.options = OptionsDict({
            'fmax': QuantityOption(1 * units.Hz, units=units.Hz, minimum=0,
                                   description='Upper cut-off frequency',
                                   help='Frequencies above this value are removed. '
                                        'Set below the cardiac frequency to suppress '
                                        'the pulse artifact.'),
            'fmin': QuantityOption(0.016 * units.Hz, units=units.Hz, minimum=0,
                                   description='Lower cut-off frequency',
                                   help='Frequencies below this value are removed. '
                                        'Set well below the slowest stimulus frequency '
                                        'to avoid attenuating the response.'),
            'butter_order': NumericOption(4, minimum=1, integer_only=True,
                                          description='Butterworth filter order',
                                          help='Order of the Butterworth filter. Higher '
                                               'values give a sharper roll-off but more '
                                               'ringing.'),
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = "Apply a band-pass filter to the input signal"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            rec[self.outputName] = freq.freq_filter(rec[self.inputName],
                                                    fmin=self.options['fmin'],
                                                    fmax=self.options['fmax'],
                                                    butter_order=self.options['butter_order'])

            return rec


class pca_filter(cedalion_module):
    # Module to apply a PCA filter to the input signal
    def __init__(self, previous_job=None):
        self.name = "PCA Filter"
        self.advanced_module = False
        self._cite = None
        self.options = OptionsDict({
            'ncomp': NumericOption(.8, minimum=0,
                                   description='Number/fraction of components removed',
                                   help='If between 0 and 1, the fraction of the variance '
                                        'removed by the filter. If >= 1, the number of '
                                        'principal components to remove.'),
            'split_types': BooleanOption(True,
                                         description='Filter each data type separately',
                                         help='If True, run the PCA filter separately for '
                                              'each wavelength / chromophore type instead '
                                              'of jointly across all of them.'),
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = "Apply a PCA filter to the input signal"
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

            rec[outputName] = pyBrainAnalyzIR.sigproc.pca_filter.pca_filter(rec[inputName],
                                                                            ncomp=self.options['ncomp'],
                                                                            split_types=self.options['split_types'])

            return rec


class ShortSeparationFilter(cedalion_module):
    # Module to regress the closest short-separation channel out of each channel
    def __init__(self, previous_job=None):
        self.name = "Short Separation Filter"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'remove_short_channels': BooleanOption(True,
                                                   description='Remove short-separation channels',
                                                   help='If True, the short-separation channels (which are flat '
                                                        'after the filter) are removed from all time series and '
                                                        'their optodes from the probe geometry.'),
            'distance_threshold': QuantityOption(DEFAULT_THRESHOLD, units=units.mm, minimum=0,
                                                 description='Short-separation distance threshold',
                                                 help='Channels with a source-detector distance below this '
                                                      'value are treated as short-separation channels.'),
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = ("Regress the closest short-separation channel out of each channel "
                            "(separately for each wavelength / chromophore)")
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
                outputName = inputName
            else:
                outputName = self.outputName

            rec[outputName] = short_separation_filter(rec[inputName], rec.geo3d,
                                                      self.options['distance_threshold'])
            if self.options['remove_short_channels']:
                remove_short_channels_recording(rec, self.options['distance_threshold'])

            return rec


# Alias matching the spelling used in the original request
ShortSeperationFilter = ShortSeparationFilter
