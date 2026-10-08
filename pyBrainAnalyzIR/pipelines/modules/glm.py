import cedalion.nirs


import cedalion.models.glm as glm
from cedalion.models.glm import basis_functions
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, BooleanOption, StringOption, QuantityOption,
    PresetOption)

units = cedalion.units


#: Commonly used hemodynamic basis sets for the GLM ``basis_function`` option.
#: The keys are what the pipeline manager GUI offers in its drop-down list;
#: from code any :class:`cedalion.models.glm.basis_functions.TemporalBasisFunction`
#: instance may be passed instead.
BasisOptions = {
    # canonical gamma variate HRF convolved with a 3 s boxcar (previous default)
    'gamma': basis_functions.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
    # gamma variate HRF plus its temporal derivative (absorbs latency shifts)
    'gamma_deriv': basis_functions.GammaDeriv(tau=0 * units.s, sigma=3 * units.s,
                                              T=3 * units.s),
    # AFNI 'GAM' response: t^p * exp(-t/q) with AFNI's default p=8.6, q=0.547 s
    'AFNIGamma': basis_functions.AFNIGamma(p=8.6, q=0.547 * units.s, T=0 * units.s),
    # deconvolution (FIR-like) model: Gaussian kernels every 3 s from -5 s to +30 s
    'GaussianKernels': basis_functions.GaussianKernels(
        t_pre=5 * units.s, t_post=30 * units.s, t_delta=3 * units.s, t_std=3 * units.s),
    # as GaussianKernels but with the first/last kernels extended as tails
    'GaussianKernelsWithTails': basis_functions.GaussianKernelsWithTails(
        t_pre=5 * units.s, t_post=30 * units.s, t_delta=3 * units.s, t_std=3 * units.s),
    # stimulus boxcar without hemodynamic convolution
    'boxcar': basis_functions.DiracDelta(),
}

DEFAULT_BASIS = 'gamma'


def BasisFunctionOption(value=DEFAULT_BASIS, **kwargs) -> PresetOption:
    """Option accepting a :data:`BasisOptions` key or any cedalion basis object."""
    kwargs.setdefault('description', 'Hemodynamic response basis')
    kwargs.setdefault(
        'help',
        'Basis function convolved with the stimulus design to model the hemodynamic '
        'response. Either the name of a predefined basis set (' + ', '.join(BasisOptions)
        + ') or, from code, any cedalion.models.glm.basis_functions object, e.g. '
        'cedalion.models.glm.Gamma(tau=0*units.s, sigma=3*units.s, T=3*units.s).')
    return PresetOption(BasisOptions, value,
                        types=basis_functions.TemporalBasisFunction, **kwargs)


def get_basis_function(basis=DEFAULT_BASIS):
    """Return the basis object for a :data:`BasisOptions` key or a basis object."""
    if isinstance(basis, str):
        return BasisFunctionOption(basis).value
    if isinstance(basis, basis_functions.TemporalBasisFunction):
        return basis
    raise TypeError('basis must be a BasisOptions key or a TemporalBasisFunction '
                    f'(got {type(basis).__name__})')


class GLM(cedalion_module):
    # Module to fit a General Linear Model (GLM) to fNIRS data
    # AR-IRLS
    # Barker, Jeffrey W., Ardalan Aarabi, and Theodore J. Huppert.
    # "Autoregressive model based algorithm for correcting motion and
    # serially correlated errors in fNIRS." Biomedical optics express 4.8 (2013): 1366-1379

    # OLS
    # Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
    # HomER: a review of time-series analysis methods for near-infrared
    # spectroscopy of the brain. Applied optics, 48(10), D280-D298.

    def __init__(self, previous_job=None):
        self.name = "GLM Model"
        self.advanced_module = False
        self.options = OptionsDict({
            'noise_model': StringOption('ar_irls',
                                        allowed=['ols', 'ar_irls', 'wls', 'gls', 'rls'],
                                        description='Noise/estimator model',
                                        help="Estimator used to solve the GLM. 'ols' is "
                                             "ordinary least-squares; 'ar_irls' is the "
                                             'autoregressive iteratively-reweighted '
                                             'least-squares model, which is robust to '
                                             'motion and serially-correlated noise.'),
            'ar_order': NumericOption(30, minimum=0, integer_only=True,
                                      description='Autoregressive model order',
                                      help='Maximum order of the autoregressive noise '
                                           "model. Only used when noise_model='ar_irls'. "
                                           'A common choice is 4x the sample rate.'),
            'max_jobs': NumericOption(1, minimum=1, integer_only=True,
                                      description='Number of parallel jobs',
                                      help='Number of channels fit in parallel. Increase '
                                           'to use more CPU cores.'),
            'basis_function': BasisFunctionOption(DEFAULT_BASIS),
            'Add_Short_Seperations': BooleanOption(
                False,
                description='Use short-separation regression',
                help='If True, the closest short-separation channel is added to the '
                     'design matrix as a nuisance regressor to remove systemic '
                     '(extracerebral) physiology.'),
            'Short_Seperation_Max_Distance': QuantityOption(
                1.5 * units.cm, units=units.cm, minimum=0, inclusive=False,
                description='Maximum short-separation distance',
                help='Source-detector distance below which a channel is considered a '
                     'short-separation channel.'),
            'drift_order': NumericOption(0, minimum=0, integer_only=True,
                                         description='Polynomial drift order',
                                         help='Order of the polynomial drift regressors '
                                              'added to the design matrix. 0 adds only a '
                                              'constant (mean) term.'),
            'verbose': BooleanOption(True,
                                     description='Print progress',
                                     help='If True, print a progress bar while the model '
                                          'is being fit.'),
        })
        self.inputName = 'conc'
        self.outputName = 'stats'
        self.description = "Fit a General Linear Model (GLM) to fNIRS data"
        self.previous_job = previous_job

    def _cite(self):
        if (self.options['noise_model'] == 'ols'):
            cite = (
                "Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009). HomER: a review "
                "of time-series analysis methods for near-infrared spectroscopy of the brain. Applied optics, "
                "48(10), D280-D298"
            )
        elif (self.options['noise_model'] == 'ar_irls'):
            cite = (
                "Barker, Jeffrey W., Ardalan Aarabi, and Theodore J. Huppert. Autoregressive model based "
                "algorithm for correcting motion and serially correlated errors in fNIRS. Biomedical optics "
                "express 4.8 (2013): 1366-1379"
            )
        else:
            cite = None
        return cite

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:

            if (self.options['Add_Short_Seperations']):
                # split time series into two based on channel distance
                ts_long, ts_short = cedalion.nirs.split_long_short_channels(
                    rec[self.inputName], rec.geo3d, self.options['Short_Seperation_Max_Distance'])
                # create design matrix from hrf and short channel regressors
                design_matrix = (glm.design_matrix.hrf_regressors(
                        ts_long, rec.stim, self.options['basis_function'])
                    & glm.design_matrix.closest_short_channel_regressor(ts_long, ts_short, rec.geo3d)
                    & glm.design_matrix.drift_regressors(ts_long, drift_order=self.options['drift_order'])
                    )
            else:
                ts_long = rec[self.inputName]
                design_matrix = (glm.design_matrix.hrf_regressors(
                            ts_long, rec.stim, self.options['basis_function'])
                    & glm.design_matrix.drift_regressors(ts_long, drift_order=self.options['drift_order'])
                )

            result = glm.fit(ts_long,
                             design_matrix=design_matrix,
                             noise_model=self.options['noise_model'],
                             max_jobs=self.options['max_jobs'],
                             verbose=self.options['verbose'])

            stats = pyBrainAnalyzIR.dataclasses.statistics.Statistics(result)
            stats.head_model = rec.head_model
            stats.geo3d = rec.geo3d
            stats.geo2d = rec.geo2d

            stats.description = "GLM model statistics from " + self.options['noise_model']

            rec[self.outputName] = stats
            return rec
