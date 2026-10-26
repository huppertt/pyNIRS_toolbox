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
    Selected named tables are combined into one output, preserving cross-ROI
    covariance. outputName defaults to <inputName>_roi unless explicitly set.
    Tests: tests/test_roi.py and tests/test_roi_statistics.py.
    """

    def __init__(self, previous_job=None, roi_tables=None):
        super().__init__(previous_job)
        self.name = "ROI Average"
        self.advanced_module = False
        self.inputName = "amp"
        self._output_name = None
        self.roi_tables = dict(roi_tables or {})
        self.selected_roi_names = list(self.roi_tables)
        self.options = OptionsDict({
            "roi_tables": ObjectOption(
                self.roi_tables, description="Named ROI membership tables",
            ),
            "roi_names": ObjectOption(
                self.selected_roi_names, description="Selected ROI names",
            ),
            "roi_table": ObjectOption(pd.DataFrame(), description="ROI membership table"),
            "roi_name": StringOption("default", description="ROI definition name"),
            "weighted": BooleanOption(
                True, description="Inverse-variance weighting",
                help="Use temporal variance for time courses or coefficient variance for statistics. "
                     "Disable for fixed table-weight pooling and OLS order comparisons. "
                     "Signed weights are normalized by their absolute sum.",
            ),
            "overwrite": BooleanOption(False, description="Replace an unused ROI definition"),
        })

    def set_roi_selection(self, tables, names):
        self.roi_tables = dict(tables)
        self.selected_roi_names = list(names)
        self.options["roi_tables"] = self.roi_tables
        self.options["roi_names"] = self.selected_roi_names

    @property
    def outputName(self):
        return self._output_name if self._output_name is not None else f"{self.inputName}_roi"

    @outputName.setter
    def outputName(self, name):
        self._output_name = None if name == f"{self.inputName}_roi" else name

    def output_names(self):
        return [self.outputName]

    def _runlocal(self, rec):
        if not isinstance(self.outputName, str) or not self.outputName.strip():
            raise ValueError("ROI output name must be a nonempty string.")
        tables = self.roi_tables or self.options["roi_tables"]
        if tables:
            names = self.selected_roi_names or self.options["roi_names"]
            if not names:
                raise ValueError("Select at least one ROI definition.")
            missing = set(names) - set(tables)
            if missing:
                raise ValueError(f"ROI tables are missing definitions: {sorted(missing)}")
            selected_tables = []
            for name in names:
                table = tables[name]
                if not isinstance(table, pd.DataFrame) or table.empty:
                    raise ValueError(f"ROI {name!r} requires a nonempty membership table.")
                table = table.assign(name=name)
                if "weight" not in table:
                    table = table.assign(weight=1.0)
                selected_tables.append(table)
            return roi_average(
                rec, pd.concat(selected_tables, ignore_index=True),
                self.inputName, self.outputName, self.outputName,
                self.options["weighted"], self.options["overwrite"],
            )
        return roi_average(
            rec, self.options["roi_table"], self.inputName, self.outputName,
            self.options["roi_name"], self.options["weighted"],
            self.options["overwrite"],
        )
