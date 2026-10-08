import cedalion.nirs
import pyBrainAnalyzIR.pipelines
import cedalion.sigproc.motion_correct as motion_correct
import pyBrainAnalyzIR.sigproc.TDDR
import pyBrainAnalyzIR.sigproc.Wavelet
import pyBrainAnalyzIR.dataclasses.dataset
import pyBrainAnalyzIR.pipelines.pipeline
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, BooleanOption, StringOption, QuantityOption)

import cedalion.sigproc.quality as quality

units = cedalion.units





class MotionCorrectPCA(cedalion_module):
    # Module to perform PCA-based motion correction on fNIRS data
    def __init__(self, previous_job=None):
        self.name = "PCA Based Motion Correction (hmrR_MotionCorrectPCA.m)"
        self.advanced_module = True
        self._cite = "Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009). HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain. Applied optics, 48(10), D280-D298."
        self.options = OptionsDict({
            'nSV': NumericOption(0.97,0.97, minimum=0,
                               description='Number of principal components to remove',
                               help='Number of principal components to remove. If a decimal between 0 and 1, it represents the fraction of variance to remove. If an integer greater than 1, it represents the exact number of components to remove.'),
            't_motion': QuantityOption(0.5,0.5, units.s,
                                      description='Time window for motion artifact detection',
                                      help='Time window for motion artifact detection.'),
            't_mask': QuantityOption(1.0,1.0, units.s,
                                    description='Time window for masking motion artifacts',
                                    help='Time window for masking motion artifacts. (+- t_mask s before/after detected motion artifact).'),
            'stdev_thresh': NumericOption(7.0, minimum=0,
                                         description='Threshold for standard deviation used to detect motion artifacts',
                                         help='Threshold for standard deviation of the signal used to detect motion artifacts. Default is 50. We set it very low to find something in our good data for demonstration purposes.'),
            'amp_thresh': NumericOption(5.0, minimum=0,
                                       description='Threshold for amplitude used to detect motion artifacts',
                                       help='Threshold for amplitude of the signal used to detect motion artifacts. Default is 5.'),
            'recursive': BooleanOption(False,
                                       description='Whether to apply recursive PCA-based motion correction',
                                       help='If True, the PCA-based motion correction will be applied recursively until no significant motion artifacts remain.'),
            'maxIter': NumericOption(10, minimum=1,
                                     description='Maximum number of recursive iterations for PCA-based motion correction',
                                     help='Maximum number of recursive iterations for PCA-based motion correction if recursive is set to True.')
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = "Perform PCA-based motion correction on fNIRS data"
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

           

            if(self.options['recursive']):
                 # to identify motion artifacts with these parameters we call the following function
                ma_mask = quality.id_motion(rec[inputName], 
                                                        self.options['t_motion'], 
                                                        self.options['t_mask'], 
                                                        self.options['stdev_thresh'], 
                                                        self.options['amp_thresh'])
                rec[outputName] = motion_correct.motion_correct_PCA(rec[inputName],
                                                                     nSV=self.options['nSV'],
                                                                     tInc=ma_mask,
                                                                     recursive=True,
                                                                     max_iter=self.options['max_iter'])
            else:
                rec[outputName] = motion_correct.motion_correct_PCA_recurse(rec[inputName],
                                                                     nSV=self.options['nSV'],
                                                                     t_motion=self.options['t_motion'],
                                                                     t_mask=self.options['t_mask'],
                                                                     stdev_thresh=self.options['stdev_thresh'],
                                                                     amp_thresh=self.options['amp_thresh'],
                                                                     maxIter=self.options['maxIter'])
            return rec



class splineSG(cedalion_module):
    # Module to perform spline-based motion correction on fNIRS data
    def __init__(self, previous_job=None):
        self.name = "Spline Based Motion Correction"
        self.advanced_module = True
        self._cite = "Molavi, B., & Dumont, G. A. (2012). Wavelet-based motion artifact removal for functional near-infrared spectroscopy. Physiological measurement, 33(2), 259-270."
        self.options = OptionsDict({
            'p': NumericOption(0.99, minimum=0, maximum=1,
                               description='Spline smoothing factor',
                               help='Smoothing factor of the spline interpolation, between '
                                    '0 (a straight line) and 1 (an interpolating spline). '
                                    'Values close to 1 follow the motion artifact closely.'),
            'frame_size': QuantityOption(10 * units.s, units=units.s, minimum=0,
                                         inclusive=False,
                                         description='Spline/Savitzky-Golay frame size',
                                         help='Length of the sliding window used by the '
                                              'Savitzky-Golay smoothing stage. Must be '
                                              'strictly positive.'),
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = "Perform spline-based motion correction on fNIRS data"
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

            rec[outputName] = motion_correct.motion_correct_splineSG(rec[inputName],
                                                                     p=self.options['p'],
                                                                     frame_size=self.options['frame_size'])
            return rec


class TDDR(cedalion_module):
    # Module to perform Temporal Derivative Distribution Repair (TDDR) motion correction on fNIRS data
    def __init__(self, previous_job=None):
        self.name = "TDDR"
        self.advanced_module = False
        self._cite = (
            'Fishburn, Frank A., Ludlum, Ruth S., Vaidya, Chandan J., and Medvedev, Andrei V. Temporal '
            'Derivative Distribution Repair (TDDR): A motion correction method for fNIRS. NeuroImage 184 '
            '(2019): 171-179.'
        )
        self.options = OptionsDict({
            'split_PosNeg': BooleanOption(True,
                                          description='Treat positive/negative shifts separately',
                                          help='If True, positive and negative temporal '
                                               'derivatives are fit separately, which '
                                               'better handles asymmetric motion artifacts.'),
            'usePCA': BooleanOption(True,
                                    description='Apply PCA before TDDR',
                                    help='If True, TDDR is applied in a PCA space shared '
                                         'across channels, which suppresses motion that is '
                                         'common to many channels.'),
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = "Perform Temporal Derivative Distribution Repair (TDDR) motion correction on fNIRS data"
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

            Fs = 1 / (rec[inputName].time[1] - rec[inputName].time[0])

            rec[outputName] = pyBrainAnalyzIR.sigproc.TDDR.TDDR(rec[inputName], Fs=Fs, usePCA=self.options['usePCA'],
                                                                split_PosNeg=self.options['split_PosNeg'])
            return rec


class Wavelet(cedalion_module):
    # Module to perform wavelet-based motion correction on fNIRS data
    def __init__(self, previous_job=None):
        self.name = "Remove Trend & Motion with Wavelets"
        self.advanced_module = True
        self._cite = None
        self.options = OptionsDict({
            'sthresh': NumericOption(5, minimum=0,
                                     description='Threshold (in std. deviations)',
                                     help='Wavelet coefficients larger than this many '
                                          'standard deviations are treated as motion '
                                          'artifact and removed.'),
            'wbasis': StringOption('sym8',
                                   allowed=['db2', 'db4', 'db8', 'sym4', 'sym8',
                                            'coif3', 'haar'],
                                   description='Wavelet basis',
                                   help='Name of the (PyWavelets) mother wavelet used for '
                                        'the decomposition.'),
            'removeScaling': BooleanOption(True,
                                           description='Remove the scaling (approximation) coefficients',
                                           help='If True, the lowest-frequency approximation '
                                                'coefficients are also removed, which detrends '
                                                'the data in addition to correcting motion.'),
        })
        self.inputName = 'last'
        self.outputName = 'last'
        self.description = "Perform wavelet-based motion correction on fNIRS data"
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

            Fs = 1 / (rec[inputName].time[1] - rec[inputName].time[0])
            rec[outputName] = pyBrainAnalyzIR.sigproc.Wavelet.Wavelet(rec[inputName], Fs=Fs,
                                                                      sthresh=self.options['sthresh'],
                                                                      wbasis=self.options['wbasis'],
                                                                      removeScaling=self.options['removeScaling'])
            return rec
