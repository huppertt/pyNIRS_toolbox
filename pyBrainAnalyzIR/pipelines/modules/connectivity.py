import cedalion.nirs

import pyBrainAnalyzIR.pipelines
import pyBrainAnalyzIR.math
from pyBrainAnalyzIR.math.corrcoef_robust import corrcoef 

import pandas as pd

from cedalion.math.ar_model import ar_filter
import numpy as np

import pyBrainAnalyzIR.pipelines.pipeline
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, BooleanOption, QuantityOption)

from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
from pyBrainAnalyzIR.math.connectivity import compute_connectivity

from pyBrainAnalyzIR.utils.geo_transforms import geo_rotation

units = cedalion.units

class resting_state_connectivity(cedalion_module):
    # Module to compute resting state connectivity from fNIRS data

    def __init__(self, previous_job=None):
        self.name = "Resting State Connectivity"
        self._cite = None
        self.options = OptionsDict({
            'AR': NumericOption(18, minimum=0, integer_only=True,
                                                  description='AR model order',
                                                  help='Use this AR model order for the connectivity computation (AR(0) skips this step ).'),
            'robust': BooleanOption(True,
                                       description='Robust estimator',
                                                 help='If True, a robust version of correlation will be used.'),
        })
        self.inputName = 'prev'
        self.outputName = 'conn'
        self.description = "Compute resting state connectivity from the input signal"
        self.previous_job = previous_job

    def _runlocal(self, rec):
        if (rec.__class__ == pyBrainAnalyzIR.dataclasses.dataset.DataSet):
            for r in rec.dataset:
                self._runlocal(r)
            return rec
        else:

            df = compute_connectivity(rec[self.inputName].copy(),
                                       robust=self.options['robust'], 
                                       AR=self.options['AR'])

            
            df['geo_to']='default'
            df['geo_from']='default'

            dataNew = Connectivity()
            dataNew.geo2d['default']=rec.geo2d
            dataNew.geo3d['default']=rec.geo3d
            dataNew.geo2d_rotation_data['default']=geo_rotation()

            dataNew.meta_data=rec.meta_data
            dataNew.head_model=rec.head_model
            dataNew.coef=df
             
            rec[self.outputName] = dataNew


        return rec
