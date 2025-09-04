import cedalion.nirs
import xarray as xr

import cedalion.models.glm as glm
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
units = cedalion.units

class GLM(cedalion_module):
    def __init__(self,previous_job=None):
        self.name = "GLM Model"
        self._cite=None
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
        
        self.previous_job = previous_job

    def _runlocal(self,rec):
        
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

        rec[self.outputName] = glm.fit(rec[self.inputName],
                                        design_matrix=design_matrix,
                                        noise_model=self.options['noise_model'],
                                        max_jobs=self.options['max_jobs'],
                                        verbose=self.options['verbose'])
                                                
        return rec
    
