"""Pipeline entry point for named ROI time-course or statistics averages."""
import pandas as pd

from pyBrainAnalyzIR.dataclasses.options_variables import (
    BooleanOption, ObjectOption, OptionsDict, StringOption,
)
from pyBrainAnalyzIR.pipelines.pipeline import cedalion_module
from pyBrainAnalyzIR.sigproc.roi import roi_average


class ROIAverage(cedalion_module):
    """Store ROI time courses or pool fitted coefficients and full covariance.

    Direct dependency: roi_average. Notebook: examples/15_ROI_Analysis.ipynb.
    Statistics weighting uses inverse coefficient variance; time courses use
    inverse temporal variance. Set weighted=False for a fixed linear average.
    Tests: tests/test_roi.py and tests/test_roi_statistics.py.
    """

    def __init__(self, previous_job=None):
        super().__init__(previous_job)
        self.name = "ROI Average"
        self.advanced_module = False
        self.inputName = "amp"
        self.outputName = "amp_roi"
        self.options = OptionsDict({
            "roi_table": ObjectOption(pd.DataFrame(), description="ROI membership table"),
            "roi_name": StringOption("default", description="ROI definition name"),
            "weighted": BooleanOption(
                True, description="Inverse-variance weighting",
                help="Use temporal variance for time courses or coefficient variance for statistics. "
                     "Disable for fixed table-weight pooling and OLS order comparisons.",
            ),
            "overwrite": BooleanOption(False, description="Replace an unused ROI definition"),
        })

    def _runlocal(self, rec):
        return roi_average(
            rec, self.options["roi_table"], self.inputName, self.outputName,
            self.options["roi_name"], self.options["weighted"],
            self.options["overwrite"],
        )
