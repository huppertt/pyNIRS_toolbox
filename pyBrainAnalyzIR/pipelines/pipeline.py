import cedalion
import cedalion.dataclasses  # noqa: F401  (namespace import)
from pprint import pprint
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, OptionVariable, BooleanOption, option_value)


def PipelineList(steps):
    """Connect module classes into an executable pipeline.

    Each item is a pipeline-module class; the function constructs each step
    with the preceding step as its input. The resulting pipeline is run with
    :meth:`cedalion_module.run`.

    Notebook examples: ``examples/03_ROC_Analysis.ipynb``.
    Direct toolbox callers: ``pyBrainAnalyzIR.testing.channelROC`` uses this
    helper to configure simulation analyses.
    Direct dependency: :class:`cedalion_module`.
    Tests: ``tests/test_analysis_pipeline.py``.
    """
    job = None
    for step in steps:
        job = step(job)

    return job


BOLD = '\033[1m'
BOLDEND = '\033[0m'
ITALICS = '\x1B[3m'
ITALICSEND = '\x1B[0m'


class cedalion_module:
    """Base class for composable analysis steps.

    Users normally construct subclasses from
    ``pyBrainAnalyzIR.pipelines.modules`` and connect them by passing the
    preceding step to the next step's constructor. Call :meth:`run` with a
    recording or :class:`~pyBrainAnalyzIR.dataclasses.dataset.DataSet` to
    execute the chain.

    Notebook examples: ``examples/02_PipelineCreation.ipynb`` and
    ``examples/06_Example_GUI_use.ipynb``.
    Direct toolbox callers: :func:`PipelineList` and
    ``pyBrainAnalyzIR.vis.NIRSviewIRWindow._run_pipeline``.
    Direct dependency: subclasses implement :meth:`_runlocal`; :meth:`run`
    invokes that method and, when present, the previous step's :meth:`run`.
    Tests: ``tests/test_pipeline_modules.py`` and
    ``tests/test_analysis_pipeline.py``.
    """
    # Advanced modules are hidden in the pipeline manager GUI unless
    # "Show Advanced Modules" is checked. Subclasses override this in __init__.
    advanced_module = True

    def __init__(self, previous_job=None):
        self.name = "default pipeline"
        self.advanced_module = True
        self._cite = None  # Citation String for the module, if applicable
        self.options = OptionsDict({
            'some_option': BooleanOption(True,
                                         description='Example option',
                                         help='Placeholder option of the base module; '
                                              'replace it in your own subclass.'),
        })
        self.inputName = 'amp'
        self.outputName = 'od'
        self.description = None  # "Write a description of what the module does"

        self.previous_job = previous_job

    """Return a string representation of the pipeline step object.
    def __repr__(self):

        if(self.previous_job is not None):
            str = self.previous_job.__repr__()
        else:
            str = ""

        str=str+f"Analysis: {self.name}\n"
        str=str+f"\t input: {self.inputName}\n"
        str=str+f"\t output: {self.outputName}\n"
        if(self._cite is not None):
            str=str+f"\t citation: {self._cite}\n"
        if(self.options is not None):
            for keys in self.options.keys:
                str=str+f"\t\t{keys} : {self.options[keys]}\n"

        return (str)
    """

    def run(self, rec):
        """Run this step and all preceding steps on a recording or dataset.

        Notebook examples: ``examples/02_PipelineCreation.ipynb`` and
        ``examples/14_Data_Quality_masking.ipynb``.
        Direct callers: notebooks and
        ``pyBrainAnalyzIR.testing.channelROC`` call this method on a module.
        Direct dependencies: the previous module's :meth:`run`, followed by
        this module's :meth:`_runlocal`.
        Tests: ``tests/test_pipeline_modules.py`` and
        ``tests/test_analysis_pipeline.py``.
        """
        # Make sure all previous jobs are run
        if self.previous_job is not None:
            rec = self.previous_job.run(rec)
        input_name, output_name = self.inputName, self.outputName
        aliases = {"last", "prev", "previous"}
        input_alias = isinstance(input_name, str) and input_name.lower() in aliases
        output_alias = isinstance(output_name, str) and output_name.lower() in aliases
        if input_alias or output_alias:
            if self.previous_job is not None:
                previous_output = self.previous_job._resolved_output_name
            else:
                recordings = getattr(rec, "dataset", [rec])
                keys = [
                    next(reversed(recording.timeseries), None)
                    for recording in recordings
                ]
                if not keys or keys[0] is None or any(key != keys[0] for key in keys):
                    raise ValueError(
                        "Cannot resolve previous pipeline data: recordings must "
                        "have a common last data key or an explicit input name."
                    )
                previous_output = keys[0]
            if previous_output is None:
                raise ValueError("The previous pipeline step has no output data name.")
            if input_alias:
                self.inputName = previous_output
            if output_alias:
                self.outputName = previous_output
        try:
            result = self._run_with_roi(rec)
            self._resolved_output_name = self.outputName
            return result
        finally:
            self.inputName = input_name
            self.outputName = output_name

    def _run_with_roi(self, rec):
        from pyBrainAnalyzIR.dataclasses.roi import get_geo_roi, propagate_roi
        import xarray as xr

        recordings = getattr(rec, "dataset", [rec])
        roi_inputs = []
        for recording in recordings:
            series = getattr(recording, "timeseries", {})
            key = next(reversed(series), None) if self.inputName == "last" else self.inputName
            data = series.get(key)
            if isinstance(data, xr.DataArray) and get_geo_roi(recording, data) is not None:
                roi_inputs.append((recording, key, data.copy(deep=False)))
        if not roi_inputs:
            return self._runlocal(rec)
        if len(roi_inputs) != len(recordings):
            raise ValueError("Pipeline input mixes ROI and physical-channel representations.")
        if getattr(self, "manages_roi_outputs", False):
            return self._runlocal(rec)
        for recording, key, source in roi_inputs:
            output_key = key if self.outputName == "last" else self.outputName
            previous_output = recording.timeseries.get(output_key)
            self._runlocal(recording)
            if output_key != key and recording.timeseries.get(output_key) is previous_output:
                continue
            # Resampling explicitly stores its new-grid masks.
            if not getattr(self, "updates_roi_masks", False):
                propagate_roi(recording, source, recording[output_key], output_key)
        return rec

    def _runlocal(self, rec):
        # Do the actual job
        return rec

    def show(self):

        if (self.previous_job is None):
            print(f"{BOLD}{self.name}:{BOLDEND}")
            if (self._cite.__class__ == self.run.__class__):
                _cite = self._cite()
            else:
                _cite = self._cite
            if (_cite is not None):
                print(f"\tCitation: {ITALICS}{_cite}:{ITALICSEND}\n")
            print("Options:")
            self.print_options()
        else:
            self.previous_job.show()
            print(f"{BOLD}{self.name}:{BOLDEND}")
            if (self._cite.__class__ == self.run.__class__):
                _cite = self._cite()
            else:
                _cite = self._cite
            if (_cite is not None):
                print(f"\tCitation: {ITALICS}{_cite}:{ITALICSEND}\n")
            print("Options:")
            self.print_options()
        return

    def print_options(self):
        """Print the options of this module (names and current values)."""
        if (not isinstance(self.options, dict)):
            pprint(self.options, sort_dicts=False)
            return
        if (len(self.options) == 0):
            print("\t<none>")
            return
        width = max(len(str(key)) for key in self.options.keys())
        for key in self.options.keys():
            opt = self.get_local_option(key)
            value = str(opt) if isinstance(opt, OptionVariable) else repr(self.options[key])
            print(f"\t{str(key):<{width}} : {value}")

    def get_local_option(self, key):
        """Return the option object for *key* from **this** module only."""
        if (isinstance(self.options, OptionsDict)):
            return self.options.option(key)
        if (isinstance(self.options, dict)):
            return self.options[key]
        return None

    def get_option(self, key):
        """Return the option object for *key* from anywhere in the pipeline.

        Unlike ``self.options[key]`` -- which returns the *current value* of this
        module's option -- this gives access to the option's default, help text
        and validation rules, and searches the previous jobs as well. The module
        closest to the end of the pipeline wins.
        """
        if (isinstance(self.options, dict) and key in self.options):
            return self.get_local_option(key)
        if (self.previous_job is not None):
            return self.previous_job.get_option(key)
        raise KeyError(f"No option '{key}' in this pipeline")

    def help(self, key=None):
        """Print the help of one option, or of every option of this module.

        When *key* is given it is looked up across the whole pipeline, so you do
        not need to know which module owns the option.
        """
        if (key is not None):
            opt = self.get_option(key)
            if (isinstance(opt, OptionVariable)):
                print(opt.format_help())
            else:
                print(f"{key}: {opt}")
            return

        if (not isinstance(self.options, dict)):
            print(f"{self.name}: no options")
            return
        print(f"{BOLD}{self.name}:{BOLDEND}")
        for k in self.options.keys():
            opt = self.get_local_option(k)
            if (isinstance(opt, OptionVariable)):
                print(opt.format_help())
            else:
                print(f"{k}: {opt}")
            print()

    def get_all_options(self):
        """Return the current *values* of the options of the whole pipeline."""
        if (self.previous_job is None):
            options = dict()
        else:
            options = self.previous_job.get_all_options()
        if (isinstance(self.options, dict)):
            for key in self.options.keys():
                options[key] = self.options[key]

        return options

    def get_all_option_objects(self):
        """Return the option *objects* of the whole pipeline, keyed by name."""
        if (self.previous_job is None):
            options = dict()
        else:
            options = self.previous_job.get_all_option_objects()
        if (isinstance(self.options, dict)):
            for key in self.options.keys():
                options[key] = self.get_local_option(key)

        return options

    def set_all_options(self, options):
        """Set the current value of any matching option in the whole pipeline.

        Values are validated by the corresponding option variable, so an invalid
        value raises rather than silently corrupting the pipeline.
        """
        if (isinstance(self.options, dict)):
            for key in self.options.keys():
                if (key in options):
                    self.options[key] = option_value(options[key])
        if (self.previous_job is not None):
            self.previous_job.set_all_options(options)

    def reset_all_options(self):
        """Restore every option of the whole pipeline to its default value."""
        if (isinstance(self.options, OptionsDict)):
            self.options.reset()
        if (self.previous_job is not None):
            self.previous_job.reset_all_options()

    def citation(self):
        if (self.previous_job is None):
            citation = []
            if (self._cite.__class__ == self.run.__class__):
                _cite = self._cite()
            else:
                _cite = self._cite
            if (_cite is not None):
                citation.append(f"{self.name} : {_cite}")
            return citation
        else:
            citation = self.previous_job.citation()
            if (self._cite.__class__ == self.run.__class__):
                _cite = self._cite()
            else:
                _cite = self._cite
            if (_cite is not None):
                citation.append(f"{self.name} : {_cite}")
            return citation
