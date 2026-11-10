# Getting Started

## What this toolbox does

pyNIRS_Toolbox (the `pyBrainAnalyzIR` Python package) provides data I/O,
composable processing and analysis pipelines, group models, connectivity
analysis, interactive GUIs, and simulation-based evaluation for fNIRS studies.
Cedalion recordings and xarray-based time series are the underlying data
objects.

## Installation

From a project checkout, install the package in editable mode:

```bash
pip install -e .
```

Use the project-supported Python environment and install Cedalion and the
declared package dependencies. For notebook examples, use a Jupyter kernel
that points to that environment. [AUTHOR: record the currently recommended
Python/Cedalion versions and platform-specific setup.]

## Typical workflow

1. Load a recording or BIDS dataset.
2. Inspect metadata, stimulus events, probe geometry, and data quality.
3. Build a pipeline from processing modules and select the input and output
   time-series names for each step.
4. Run and inspect first-level results; for a multi-recording study, add
   demographics and run an appropriate group model.
5. Use the statistics or connectivity viewer to inspect maps, model terms,
   result tables, and fit plots.
6. Save/export results using the relevant BIDS or table-export workflow.

## Minimal pipeline pattern

```python
from pyBrainAnalyzIR.pipelines.modules import preproccessing as prep
from pyBrainAnalyzIR.pipelines.modules.glm import GLM

job = prep.intensity_opticaldensity()
job = prep.mbll(job)
job = GLM(job)
job.run(recording)
```

Set and verify each module's `inputName`, `outputName`, and options for the
actual input data. The example is illustrative; it is not a universal
preprocessing recipe.

## Where to go next

- [Data Model and I/O](Data-Model-and-IO)
- [Pipelines and Modules](Pipelines-and-Modules)
- [Examples and Demos](Examples-and-Demos)

## Maintainer prompts

- [AUTHOR: add environment creation instructions and dependency installation.]
- [AUTHOR: describe how users should report issues or request support.]
