import cedalion
import cedalion.dataclasses  # noqa: F401  (namespace import)
from pprint import pprint
from pyBrainAnalyzIR.dataclasses.options_variables import (
    OptionsDict, OptionVariable, BooleanOption, option_value)


def PipelineList(steps):
    job = None
    for step in steps:
        job = step(job)

    return job


BOLD = '\033[1m'
BOLDEND = '\033[0m'
ITALICS = '\x1B[3m'
ITALICSEND = '\x1B[0m'


class cedalion_module:
    def __init__(self, previous_job=None):
        self.name = "default pipeline"
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
        # Make sure all previous jobs are run
        if (self.previous_job is None):
            # This is the first module on a pipeline run it
            return self._runlocal(rec)
        else:
            return self._runlocal(self.previous_job.run(rec))

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
