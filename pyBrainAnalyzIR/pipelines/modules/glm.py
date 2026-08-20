import cedalion.nirs
import xarray as xr

import cedalion.models.glm as glm
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR
units = cedalion.units
import pyBrainAnalyzIR.dataclasses.dataset

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

    def __init__(self,previous_job=None):
        self.name = "GLM Model"
        self.options={'noise_model':'ols',
                      'ar_order':30,
                      'max_jobs':1,
                      'basis_function':cedalion.models.glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
                      'Add_Short_Seperations': False,
                      'Short_Seperation_Max_Distance':1.5*units.cm,
                      'drift_order':0,
                      'verbose':True}
        self.inputName='conc'
        self.outputName='stats'
        self.description="Fit a General Linear Model (GLM) to fNIRS data"
        self.previous_job = previous_job

    def _cite(self):
        if(self.options['noise_model']=='ols'):
            cite="Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009). HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain. Applied optics, 48(10), D280-D298"
        elif(self.options['noise_model']=='ar_irls'):
            cite="Barker, Jeffrey W., Ardalan Aarabi, and Theodore J. Huppert. Autoregressive model based algorithm for correcting motion and serially correlated errors in fNIRS. Biomedical optics express 4.8 (2013): 1366-1379"
        else:
            cite=None
        return cite




    def _runlocal(self,rec):
        if(rec.__class__==pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:
            
            if(self.options['Add_Short_Seperations']==True):
                # split time series into two based on channel distance
                ts_long, ts_short = cedalion.nirs.split_long_short_channels(
                    rec[self.inputName], rec.geo3d, self.options['Short_Seperation_Max_Distance'])
                # create design matrix from hrf and short channel regressors
                design_matrix = ( glm.design_matrix.hrf_regressors(
                        ts_long, rec.stim, self.options['basis_function'])
                    & glm.design_matrix.closest_short_channel_regressor(ts_long, ts_short, rec.geo3d)
                    & glm.design_matrix.drift_regressors(ts_long, drift_order=self.options['drift_order'])
                    )
            else:
                ts_long=rec[self.inputName]
                design_matrix = (glm.design_matrix.hrf_regressors(
                            ts_long, rec.stim, self.options['basis_function'])
                    & glm.design_matrix.drift_regressors(ts_long, drift_order=self.options['drift_order'])
                )

            result = glm.fit(ts_long,
                                    design_matrix=design_matrix,
                                    noise_model=self.options['noise_model'],
                                    max_jobs=self.options['max_jobs'],
                                    verbose=self.options['verbose'])
            
            stats=pyBrainAnalyzIR.dataclasses.statistics.Statistics(result)
            stats.head_model=rec.head_model
            stats.geo3d=rec.geo3d
            stats.geo2d=rec.geo2d

            stats.description="GLM model statistics from " + self.options['noise_model']

            rec[self.outputName]=stats
            return rec
    
