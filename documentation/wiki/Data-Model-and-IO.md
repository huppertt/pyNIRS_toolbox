# Data Model and I/O

## Core objects

### Cedalion recording

A recording stores named time series, stimulus events, probe geometry, metadata,
auxiliary channels, and masks. Time-series arrays are generally labeled
`xarray.DataArray` objects with dimensions/coordinates such as `channel`,
`chromo` (or wavelength), and `time`; units may be represented with Pint.
Available named series depend on which pipeline steps have been run.

### `DataSet`

`pyBrainAnalyzIR.dataclasses.dataset.DataSet` groups recordings and stores
dataset-level statistics and demographics. Recordings are in `dset.dataset`;
group results are available by name from `dset.statistics` or
`dset["result_name"]`.

```python
from pyBrainAnalyzIR.dataclasses.dataset import DataSet

dset = DataSet()
dset.import_data(recording)
dset.add_demographics_by_index(demographics)
```

Subject identifiers and row order in demographics must correspond to the
recordings as required by the chosen group-analysis method. Prefer explicit
subject IDs and verify their alignment.

### Stimulus and metadata tables

Stimulus/event tables describe condition labels and event timing (commonly
`onset`, `duration`, and amplitude). Recording metadata carries subject and
study fields; demographics are tabular and are used by group models.

### Analysis result objects

- `Statistics` stores channel/type/condition estimates, covariance and geometry;
  modern results may also retain model-fit data for the viewer.
- `Connectivity` stores edge-level coefficients, p/q-values, endpoint
  identities/types, and geometry. Hyperscanning results may identify both
  participant endpoints.
- `GeoROI` and ROI tables describe weighted region membership and ROI signals.
- `FirstLevelFit`, `GroupModelFit`, and `ConnectivityGroupFit` retain fit or
  observation details used by the interactive viewers.

## Supported I/O

The `pyBrainAnalyzIR.io` package currently exposes:

```python
from pyBrainAnalyzIR.io import loadDirectory, loadDotNIRS, loadNIRx
```

- `loadDotNIRS`: import DotNIRS `.nirs` files.
- `loadNIRx`: import supported NIRx recordings and sidecars.
- `loadDirectory`: recursively assemble a dataset, optionally grouping folder
  hierarchy into metadata.
- BIDS helpers in `pyBrainAnalyzIR.io.bids` read/write supported BIDS-NIRS
  datasets and associated metadata.
- SNIRF read/write is used by Cedalion and the BIDS integrations.

Format support can depend on sidecars and exporter conventions. Validate
stimulus timing, wavelength/chromophore labels, units, demographics, and probe
positions after importing a study.

## User checks

- Confirm sample/time units and channel/type coordinates before analysis.
- Confirm each recording has the expected event definitions and probe geometry.
- Review data masks and missing channels before fitting models.
- Preserve a copy of the unmodified source data; pipelines may add or update
  named time series in place.

## Maintainer prompts

- [AUTHOR: document exact field conventions used by the lab's supported file
  formats.]
- [AUTHOR: add supported BIDS/SNIRF version notes and a tested import checklist.]
