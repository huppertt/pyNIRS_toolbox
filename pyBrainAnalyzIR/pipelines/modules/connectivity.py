import cedalion.nirs

import pyBrainAnalyzIR.pipelines
import pyBrainAnalyzIR.math


import numpy as np

import pyBrainAnalyzIR.pipelines.pipeline
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module as cedalion_module
import pyBrainAnalyzIR.dataclasses.dataset
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, NumericOption, BooleanOption, StringOption)

from pyBrainAnalyzIR.dataclasses.connectivity import Connectivity
from pyBrainAnalyzIR.math.connectivity import compute_connectivity, compute_hyperscanning

from pyBrainAnalyzIR.utils.geo_transforms import geo_rotation

units = cedalion.units


class resting_state_connectivity(cedalion_module):
    # Module to compute resting state connectivity from fNIRS data

    def __init__(self, previous_job=None):
        self.name = "Resting State Connectivity"
        self._cite = "Santosa, Hendrik, et al. 'Characterization and correction of the false-discovery rates in resting state connectivity using functional near-infrared spectroscopy.' Journal of biomedical optics 22.5 (2017): 055002-055002."  # noqa: E501
        self.options = OptionsDict({
            'AR': NumericOption(18, minimum=0, integer_only=True,
                                description='AR model order',
                                help='Use this AR model order for the connectivity computation (AR(0) skips this step ).'),  # noqa: E501
            'robust': BooleanOption(True,
                                    description='Robust estimator',
                                    help='If True, a robust version of correlation will be used.'),
        })
        self.inputName = 'last'
        self.outputName = 'conn'
        self.description = "Compute resting state connectivity from the input signal"
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

            df = compute_connectivity(rec[inputName].copy(),
                                      robust=self.options['robust'],
                                      AR=self.options['AR'])

            df['geo_to'] = 'default'
            df['geo_from'] = 'default'

            dataNew = Connectivity()
            dataNew.geo2d['default'] = rec.geo2d
            dataNew.geo3d['default'] = rec.geo3d
            dataNew.geo2d_rotation_data['default'] = geo_rotation()

            dataNew.meta_data = rec.meta_data
            dataNew.head_model = rec.head_model
            dataNew.coef = df

            dataNew = dataNew.ttest(remove_cross_types=True)

            rec[self.outputName] = dataNew

        return rec


class hyperscanning(cedalion_module):
    # Module to compute hyperscanning fNIRS data

    def __init__(self, previous_job=None):
        self.name = "Hyperscanning Connectivity"

        self._cite = "Santosa, Hendrik, et al. 'Characterization and correction of the false-discovery rates in resting state connectivity using functional near-infrared spectroscopy.' Journal of biomedical optics 22.5 (2017): 055002-055002."  # noqa: E501

        self.options = OptionsDict(
            {
                'AR': NumericOption(
                    18,
                    minimum=0,
                    integer_only=True,
                    description='AR model order',
                    help='Use this AR model order for the connectivity computation (AR(0) skips this step ).'),
                'robust': BooleanOption(
                    True,
                    description='Robust estimator',
                    help='If True, a robust version of correlation will be used.'),
                'grouping_variable': StringOption(
                    'pair',
                    description='Dyad grouping variable',
                    help='The name of the variable in the demographics that indicates which subjects are in a dyad. This is used to compute hyperscanning connectivity between subjects in the same dyad.'),  # noqa: E501
                'unordered': BooleanOption(
                    False,
                    description='Treat as unordered pairs',
                    help='if false, then the order of the subjects in the dyad will be used to determine which subject is "from" and which is "to". If true, then the subjects will be treated as unordered pairs.'),  # noqa: E501
            })
        self.inputName = 'last'
        self.outputName = 'hyperconn'
        self.description = "Compute resting state connectivity from the input signal"
        self.previous_job = previous_job

    def _runlocal(self, dset):
        demo = dset.get_demographics()

        if (self.inputName == 'last'):
            inputName = list(dset.dataset[0].timeseries.keys())[-1]
        else:
            inputName = self.inputName

        outtype = self.outputName
        grouping_variable = self.options['grouping_variable']

        upairs = demo[grouping_variable].unique()
        for upair in upairs:
            print(f'Computing connectivity for {grouping_variable}={upair}')
            lst = np.where(demo[grouping_variable] == upair)[0]
            dyad = {}
            if len(lst) > 1:
                for i in lst:
                    dyad[demo['subject'][i]] = dset.dataset[i][inputName]
                df = compute_hyperscanning(
                    dyad,
                    robust=self.options['robust'],
                    AR=self.options['AR'],
                    unordered=self.options['unordered'])
                df['geo_to'] = df.Subject_to
                df['geo_from'] = df.Subject_from

                rotations = np.linspace(0, 2 * np.pi, len(lst), endpoint=False)
                rotations = rotations - rotations.mean()

                dataNew = Connectivity()
                for idx, i in enumerate(lst):
                    name = demo['subject'][i]
                    dataNew.geo2d[name] = dset.dataset[i].geo2d
                    dataNew.geo3d[name] = dset.dataset[i].geo3d

                    geo_rot = geo_rotation()
                    geo_rot.set_rotation2D(np.array([rotations[idx]]))
                    geo_rot.translation2D = np.array([100 * np.sin(rotations[idx]),
                                                      -100 * np.cos(rotations[idx])])
                    dataNew.geo2d_rotation_data[name] = geo_rot

                    # TODO: Add meta_data and head_model for each subject in the dyad
                    # dataNew.meta_data=dset.meta_data
                    # dataNew.head_model=dset.head_model

                dataNew.coef = df
                dataNew = dataNew.ttest(remove_same_geo=True, remove_cross_types=True)
                for i in lst:
                    dset.dataset[i][outtype] = dataNew

        return dset
