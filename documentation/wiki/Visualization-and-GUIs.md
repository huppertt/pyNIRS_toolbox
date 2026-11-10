# Visualization and GUIs

## NIRSviewIR

NIRSviewIR is the integrated session browser. It supports recording selection,
time-series/probe inspection, stimulus and metadata workflows, pipeline
management, and embedded result views. Selecting a first-level Statistics or
Connectivity result, or a dataset-level `Group: ...` result, opens the
corresponding shared viewer inside the application. Returning to a time-series
restores the time-course view.

## Statistics viewer

`StatisticsViewer` / `view_statistics` display first-level or group-level
`Statistics` objects. The map can be filtered by model term/condition, signal
type, and p- or q-value threshold. Detail plots show first-level fits or
group-level term-specific plots; the results table supports significance
highlighting, sorting, clipboard copy, and Excel export.

```python
from pyBrainAnalyzIR.vis import view_statistics

viewer = view_statistics(dset["groupstats"])
```

## Connectivity viewer

`ConnectivityViewer` / `view_connectivity` display first-level connectivity,
hyperscanning, and group connectivity results. Controls select task/model term,
signal-type pair, threshold type/value, and (for hyperscanning) inter-/intra-
participant endpoints. Maps include probe and adjacency-matrix views where
available. Results are filtered for significance by default; the Results Table
tab can show all edges, highlight significant rows, and copy/export the table.
The Results Plot tab shows time courses/scatter for first-level results or
model-term plots for group results.

```python
from pyBrainAnalyzIR.vis import view_connectivity

viewer = view_connectivity(dset.dataset[0]["conn"], data=dset, input_name="conc")
```

## Other visualization and workflow windows

- **Pipeline Manager:** discover modules, edit options and order, import/export
  pipeline definitions.
- **Stimulus manager:** review and edit event annotations.
- **Data quality/masking managers:** inspect and edit signal-quality masks.
- **Demographics manager:** view and edit participant-level variables.
- **Registration tools:** align probe/head geometry and manage head-model
  registrations.
- **ROI manager and ROI plots:** define regions, inspect their channel
  composition, and visualize ROI estimates.
- **Inline plots:** probe, time-series, and ROI plotting helpers for notebooks.

Some tools are launched from NIRSviewIR or module-specific functions; consult
the GUI and examples for exact entry points.

## Maintainer prompts

- [AUTHOR: add screenshots and a GUI walkthrough.]
- [AUTHOR: document keyboard shortcuts, context menus, and known platform
  differences.]
