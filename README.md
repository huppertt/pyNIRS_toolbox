# pyBrainAnalyzIR / pyNIRS Toolbox

[![Tests](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/tests.yml/badge.svg)](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/tests.yml)
[![Pylint](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/pylint.yml/badge.svg)](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/pylint.yml)
[![Flake8](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/flake8.yml/badge.svg)](https://github.com/huppertt/pyNIRS_toolbox/actions/workflows/flake8.yml)

A Python wrapper for the Cedalion fNIRS analysis framework, providing high-level pipelines, GUI tools, and statistical models for functional near-infrared spectroscopy data.

## Package Information

| Field        | Value                               |
|--------------|-------------------------------------|
| **Name**     | pyBrainAnalyzIR                     |
| **Version**  | 0.1.0                               |
| **Author**   | Theodore Huppert                          |
| **Affiliation** | Stony Brook University, Department of Psychiatry and Behavioral Health |
| **Build Date** | 2026-08-20                        |
| **License**  | BSD 3-Clause                        |
| **Repository** | https://github.com/huppertt/pyNIRS_toolbox |

## Description

pyBrainAnalyzIR provides:
- A modular pipeline system (`pyBrainAnalyzIR.pipelines`) for building fNIRS preprocessing and analysis workflows.
- GUI tools (`pyBrainAnalyzIR.vis`) for interactive data visualization and pipeline management.
- Statistical models for first-level and group-level GLM analysis.
- Compatibility with the [cedalion](https://github.com/ibs-lab/cedalion) data format and processing engine.

## Installation

```bash
pip install -e .
```

## Requirements

- Python ≥ 3.10
- PySide6
- cedalion
- numpy, scipy, pandas, xarray, matplotlib

## Interactive statistics viewer

Fit the first-level GLM and/or group-level MixedEffects model, then open a
single statistics object:

```python
from pyBrainAnalyzIR.vis import view_statistics

# In a notebook; keep the window reference.
viewer = view_statistics(dset["groupstats"])
# Or: viewer = view_statistics(rec["stats"])
# In a standalone script, pass block=True.
```

NIRSviewIR embeds this same viewer when you select a recording's Statistics
variable or a `Group: ...` result in the Data type selector. The recording
browser stays available; switching to a timeseries restores the timecourse
display. Connectivity results retain their separate connectivity interface.

When a pipeline is launched from NIRSviewIR, its Python stdout/stderr output
(including print messages, warnings, and text progress bars) is also mirrored
into the status bar. Terminal/notebook output is retained. Routing is scoped
to that synchronous GUI run and restored even on failure; unrelated threads
and subprocess output are not redirected into the GUI.
GLM fitting also reports completed channel/signal fits directly to a status-bar
progress widget, independently of tqdm streams and the GLM `verbose` setting.
Progress includes both serial and parallel fits. GUI paint events are serviced
during updates, with user-input events excluded while the pipeline runs.

Select a condition/model term and signal type to see its t-statistic map.
Choose a p-value or Benjamini-Hochberg q-value threshold (q-values are corrected
over all finite estimates in the statistics object). Non-significant channels
are gray but remain clickable; the channel dropdown also selects a channel.
Without probe geometry, a channel-index map is shown instead. When the
statistics include a `geo3d` probe with Nz/LPA/RPA head landmarks, the map
opens as a 10-20 view; right-click empty probe space to switch between the
10-20 view and the 2D probe (`view_statistics(..., plot_on_scalp=False)` starts
in 2D).

The right panel has two tabs. **Results Plots** shows the model-fit plots below.
**Results Table** lists every signal type for the selected condition/model term;
rows passing the current threshold are highlighted and marked in a
`Significant` column. Selecting a row selects that channel and signal. Right-click
the table to **Copy to Clipboard** (tab-delimited with a header row, ready to
paste into Excel) or **Save to Excel...** (`.xlsx`).

Clicking a first-level channel displays the original data and full model
prediction, including nuisance terms. Quality-masked samples are excluded.
The panel reports descriptive original-space R² and a joint covariance-based
Wald F-test of the nonconstant regressors. The F-test uses the estimator's
coefficient covariance and residual/effective degrees of freedom; R² is not
a whitened fit statistic.

For group results, **Condition/Model term** determines both the statistical map
and the detail plot. No separate observed-task, x-axis or adjustment controls
are needed. A condition intercept such as `Condition[HRF A]` shows a box plot;
`Condition[HRF A]:age` shows a scatter plot against age. Categorical effects show
category box plots. Multivariable interactions use the selected design-column
value as the x-axis rather than choosing an arbitrary component.

All other fixed-effect contributions are **always subtracted** from both subject
betas and individual fits. For selected design column `x_j`, the plotted beta is
`Beta - (X beta - x_j beta_j)` and the population curve is `x_j beta_j`.
For `Beta ~ 0 + Condition + Condition:age`, selecting the age slope therefore
removes the condition intercept and shows `beta_age * (age - training_mean)`;
selecting the condition intercept removes the age contribution instead.
Transformed variables use the fitted Patsy coding and training centers.
Single-variable x-axes retain original demographic units; a centered slope
crosses zero at the training mean, not at raw age zero.

Plots stay in original, unwhitened beta units, retaining residuals and random
effects. Subject points have two error bars:

- **Blue: ±1 first-level SE**, the square root of the corresponding diagonal of
  the input first-level covariance.
- **Purple: ±1 final weight-implied SD**, mapped back into original beta units.
  If `W` whitens the first-level covariance and `D` contains the solver's final
  robust row multipliers, the implied working covariance is
  `W^-1 D^-2 W^-T`. Purple bars use the square root of its diagonal. The solver
  multiplies the response and both designs by `D`, so precision uses `D²`.
  Full-covariance whitening mixes observations: this is not generally
  `first_level_SE / robust_weight`, and a whitened multiplier is not a scalar
  weight belonging to one original subject point.

With robust fitting disabled, weighted fits have coincident blue/purple bars.
Downweighting increases the weight-implied marginal uncertainty. Zero
multipliers imply unbounded uncertainty in affected directions; affected
points remain visible with an explicit message, without a finite purple bar.
Unweighted fits use `W = I`, so the purple baseline is unit variance rather
than the first-level covariance and need not exceed the blue bars.
Purple bars describe the final observation-weighting metric, **not calibrated
95% confidence intervals**. They do not include the fitted residual scale,
random-effect variance, or uncertainty in subtracting other fixed effects;
neither set of point bars is a subject prediction interval. Marginal bars
cannot show the full correlation structure used in the fit.

`Statistics.fit_data.observations` retains `WeightStdErr` and
`WhitenedRobustWeight` (the row multiplier before mapping back).
Rerun MixedEffects to populate these for older results; the viewer explicitly
reports missing final weights rather than inventing them.

Red X's always show the corresponding partial
individual fits, alongside two pointwise 95% confidence bands/whiskers:

- **Red: selected-term population mean.** Variance is `x_j² Cov(beta)[j,j]`.
- **Orange: full fixed-effects mean (translated).** Variance is
  `x Cov(beta) x.T`, including the intercept, other coefficients and all their
  cross-covariances. For intercept plus slope this is
  `Var(intercept) + 2*z*Cov(intercept,slope) + z²*Var(slope)`.
  The full-mean interval is translated to the partial-fit line by subtracting
  the fitted nuisance offset as a fixed display transformation. It is not a
  propagated confidence interval for the adjusted observations themselves.
  Other demographics are held at training mean/modal values; multivariable
  interactions use observed predictor combinations, ordered by design value.

The orange interval includes uncertainty omitted by the selected-term interval,
but cross-covariance can make it narrower at some predictor values; widths are
not artificially forced to nest. Both describe uncertainty in a fitted mean,
**not a subject prediction interval**: neither includes random-effect/residual
variation or is expected to contain 95% of subject points. Large subject scatter
can coexist with a precise population mean.

Category points are
horizontally jittered; gray box plots summarize **unweighted** partial subject
betas (median and middle 50%), not a fitted mean or confidence interval.
A summary above the explanatory text reports each category's median,
fitted mean, fit-minus-median gap, weighting mode and robust setting. Opposite
signs are explicitly flagged for covariance/robust sensitivity inspection.
Full-covariance mixed-model influence is not generally a positive scalar
weighted average of the displayed points; marginal blue/purple bars do not
explain signed cross-channel/condition influence. A large median/fit gap is
neither proof of invalid inference nor reassurance that inference is valid.
There is one
point per subject for a task when there is one recording per subject; repeated
recordings remain separate observations. Coefficients shared across tasks pool
their observation rows after removing each row's other fixed effects.
Plot navigation toolbars are omitted; control labels are right-aligned for spacing.

The GLM and MixedEffects outputs now retain `Statistics.fit_data`. Rerun models
created with older toolbox versions to populate it; older results still support
the statistical map. Removing conditions or applying display contrasts does not
change the retained full model fit. Newly constructed display contrasts that
are not original fitted coefficients show an explicit partial-plot-unavailable
message instead of an unrelated model curve.
Pooling existing statistics into ROIs clears channel-fit details; fit the GLM
on the ROI time series to inspect ROI time courses, or run MixedEffects on the
pooled first-level statistics to inspect group ROI fits.

### Group-model covariance weighting

`MixedEffects` defaults to `weighted=True` and `covariance_weighting="full"`.
Full whitening requires numerically positive-definite first-level covariances.
Indefinite or numerically singular matrices now raise an actionable error:
silently replacing negative eigenvalues with tiny positive values can assign
extreme precision to unsupported directions and severely distort group fits.
Existing group estimates are not retroactively corrected.

Full-covariance whitening defaults to lower Cholesky. An opt-in SVD basis is
available without changing the covariance model or the shared-subject random
effects:

```python
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects

job = MixedEffects()
job.options["whitening_method"] = "svd"  # default: "cholesky"
```

For `C = U diag(d) U.T`, SVD uses `W = diag(1 / sqrt(d)) @ U.T`.
Both bases satisfy `W @ C @ W.T = I` and retain cross-channel/condition
covariance. Nonrobust GLS is mathematically unchanged, although numerical
variance optimization can differ. Componentwise robust fitting is basis-dependent:
SVD can change robust estimates and uncertainties and is not guaranteed to
improve outlier performance or p-value calibration. Eigenvector signs and
orientations within repeated-singular-value subspaces are not unique; SVD is
not a universal invariance fix. Both methods reject invalid full covariances,
without eigenvalue flooring. The selected basis is retained in covariance and
fit-observation metadata. `whitening_method` is ignored with diagonal weighting
or `weighted=False`; metadata marks it `not_applicable`.

An explicit alternative is inverse-variance-only weighting:

```python
from pyBrainAnalyzIR.pipelines.modules.mixedeffects import MixedEffects

job = MixedEffects()
job.options["FE_formula"] = "Beta ~ 0 + Condition + Condition:age"
job.options["covariance_weighting"] = "diagonal"
dset = job.run(dset)
```

This uses first-level coefficient variances while ignoring their cross-covariances;
it is a different weighting model, not an automatic covariance repair.
Alternatively, `weighted=False` fits without first-level covariance weighting.
Neither choice changes which covariates are included in the formula.

### Experimental outer covariance-inflation IRLS

`pyBrainAnalyzIR.testing.mixed_effects_irls.covariance_inflation_irls` is an
isolated test function; it does **not** change `MixedEffects` or its defaults.
It takes unwhitened `y`, fixed design `x`, random design `z`, and a list of
first-level covariance blocks. Stack all three arrays **subject/recording
first**, with each block's rows in exactly the same channel/condition/type
order as its measurements and designs. Blocks may have different sizes.
Construct the full channel-specific fixed design before calling the function;
no formula expansion or automatic reordering happens inside it.

```python
from pyBrainAnalyzIR.testing.mixed_effects_irls import covariance_inflation_irls

trial = covariance_inflation_irls(
    y, x, z, subject_covariances, max_iterations=100, tolerance=1e-5,
)
print(trial.converged, len(trial.history))
print(trial.beta, trial.gamma)
print(trial.inflation, trial.precision_weights)
```

The loop starts with `s = 1`, explicitly rebuilds each
`W_i = solve(cholesky(diag(s_i) @ C_i @ diag(s_i)), I)` using lower Cholesky,
and stacks `W_all = block_diag(W_i)`. Every solve applies the **same**
`W_all` to `y`, `x` and `z`, with no inner robust fit. Original conditional
residuals `y - x @ beta - z @ gamma` update a separate factor for every
subject/channel/condition/type observation.

Here `s` is an **uncertainty-inflation** factor (at least 1), not a small
robust precision weight. The test uses positive Huber precision weights
`q = min(1, 1.345 / abs(u))` and `s = 1 / sqrt(q)`, avoiding singular
covariances from zero bisquare weights. `u` divides the original residual
by its original first-level marginal SE and then by a pooled MAD scale.
Each iteration starts from the **original** covariance, not the covariance
inflated in the previous iteration. No damping or leverage correction is
applied. Weights, fixed/random coefficients and the log variance ratio must
all stabilize to declare convergence.

The experiment reuses the nonrobust `solveLME` solve with a bounded ML scalar
variance search (`theta` in [-25, 20]); boundary estimates warn. Iteration
limits warn and return `converged=False`. Invalid covariance/inputs or
undefined residual MAD raise explicit errors. Returned whitening/inflation
always correspond to the returned fitted coefficients, including cap exits.
Inputs are not mutated.

`trial.covariance` is **conditional working-model** coefficient covariance,
not validated robust confidence intervals. This residual rule can still
mask artifacts absorbed by subject random effects; it is a test of the
requested loop, not a claim that bias is fixed. It does not create group
`Statistics` p-values or alter the statistics viewer.

### Shared subject random effects

`MixedEffects` now defaults to **one subject intercept shared across all
channels, chromophores, conditions, and recordings of that subject**:

```python
job = MixedEffects()
job.options["RE_formula"] = "subject"  # recommended default
job.options["group_by"] = "subject"    # metadata identifying a person
```

The default `group_by="auto"` uses `subject`, then `subjectID`. If neither
column exists it warns and uses separate recording groups. Specify another
metadata column for other subject-ID conventions, or `"fileIdx"` to explicitly
group by recording. Missing/empty IDs are rejected; numeric IDs are not
mean-centered. Repeated recordings share the same random coefficient and do
not increase the subject count used for group residual degrees of freedom.

The original-space model is `beta[s,c,h,k] = X[s,c,h,k] b + u[s] + error[s,c,h,k]`.
Both the fixed and random designs are constructed before whitening; the same
first-level covariance whitener is applied to response, fixed design, and
random design. Thus unequal measurement uncertainty does not change the
meaning of the shared offset in original beta units. The statistics GUI's
conditional fits retain this same shared offset in unwhitened units.

Every random-effects term follows the same rule: **only factors named in that
term split its coefficients**. Factors are categorical grouping identifiers,
including numeric IDs; this syntax does not specify continuous random slopes.
An optional leading `~` is accepted. `cond`, `channel`, and `chromo` alias the
statistics table's `Condition`, `Channel`, and `Type`. Other demographic columns
can be used directly, for example `subject + site` adds a globally shared
site effect, not a separate site effect for every subject or channel.

For 12 subjects, 16 channels, two chromophores, and two conditions:

| `RE_formula` | Random coefficients | Shared across |
|---|---:|---|
| `subject` (recommended) | 12 | Conditions, channels, chromophores |
| `subject:cond` | 24 | Channels, chromophores |
| `subject:cond:channel` | 384 | Chromophores |
| `subject:cond:channel:chromo` | 768 | No additional dimensions |

Interaction terms use one indicator per observed factor combination, not
reference-level contrasts; unobserved combinations do not create empty columns.
`+` adds separate random-effect blocks (for example `subject + subject:cond`).
The existing low-level solver uses a single isotropic random-effect variance
parameter across coefficients; the formula controls sharing, not separate
variance components for each term. Intercept-only `~1` is rejected: name
`subject` explicitly instead. Earlier `~Condition` syntax no longer implicitly
introduces subjects/channels; migrate to the explicit factor formula.

Result covariance attributes record the formula, each term's coefficient count,
grouping column, subject count, and total random-coefficient count.
Refit saved group results to use the new
model. Historical ROC studies below used the previous channel-specific random
effects unless explicitly labeled shared-subject.

This is a substantive biological covariance assumption, not a consequence of
whitening alone: first-level covariance describes measurement uncertainty;
a shared subject intercept describes an additional between-subject shift.
It does not represent every possible channel-specific biological difference,
so raw-p calibration must still be checked for the intended application.

On the previously failing ordinary-length IID dataset, the shared model
reduced random-effect count from 96 to 12. One null channel's age-effect SE
changed from `2.61e-7` to `0.0852` and its raw p-value from `3.34e-47` to
`0.825`. The same first-level covariance and fixed design were used.

A separate known-covariance experiment generated 100 independent datasets,
12 subjects, four signals, and two conditions with a true shared subject
intercept (SD 0.06), heterogeneous known measurement SDs, and measurement
correlation 0.25. All population effects were null:

| Group term | Nonrobust raw FPR | Robust raw FPR |
|---|---:|---:|
| Mean | 5.13% | 2.88% |
| Age | 6.88% | 4.13% |

Recording/dataset bootstrap intervals contained 5% in all four cases; no
conditional-scale-collapse warning occurred. This validates the requested
structure under that data-generating model, not arbitrary biological
heterogeneity. Channel/ROI Monte Carlo checks also retain the older,
channel-specific biological simulation to assess model-mismatch behavior.

Those channel/ROI checks ran 80 independent datasets (20 each for IID, AR,
active AR + motion, and all-null AR + motion), twelve subjects per dataset:
2,880 first-level GLMs and 9,600 group fits, with no inference/fitting failures
or flagged conditional-scale collapse. The active-regime channel robust-group
raw FPR was 0% for mean/age/categorical terms, versus 40-100% for several old
channel-specific fits. These new fits were conservative, not proof of exact
5% calibration; robust age sensitivity was 70-96.25% depending on regime.

All-null AR + motion remains less well calibrated: shared-model robust-group
channel FPR was 6.88% for the mean and 5.63% for age/categorical terms. Mean
FPR across channel/ROI paths ranged from 6.88% to 16.25%, with broad
twenty-dataset intervals. Thus changing the random design removes the
catastrophic saturation failure but does not establish universal group/ROI
calibration. The active batch used the interim shared-intercept `~1` notation;
the final `subject` formula produces the identical original-space indicator
design, verified independently before and after whitening. The all-null
batch and known-covariance experiment used the final formula implementation.

### Interim AR-IRLS joint covariance repair

AR-IRLS uses robust AR coefficient regression by default; see
[Robust AR coefficient estimation](#robust-ar-coefficient-estimation).
Historical calibration studies below used the original OLS AR step unless
explicitly identified as robust-AR studies.

AR-IRLS regularizes its **cross-signal covariance approximation** before
first-level statistics are exposed. The old pairwise robust residual estimator
and inverse cross-design products do not, in general, define a PSD matrix.
The assembled diagonal blocks differ from each individual fit's `cov_params()`,
but replacing them proved to worsen FDR. The current patch deliberately
preserves the **legacy assembled diagonal blocks**, not individual-fit
`cov_params()`. The covariance repair itself does not change the AR-IRLS coefficient or marginal
uncertainty estimator. Changing the AR coefficient estimator does change fitted
coefficients and uncertainty; block preservation compares raw and repaired
covariance from the **same** fitted estimator.

The implementation in [`covariance.py`](pyBrainAnalyzIR/math/glm/covariance.py):

1. Retains each signal's entire **legacy joint-covariance diagonal block**,
   including within-signal regressor covariances, not just its variances.
   Only cross-signal blocks may change.
2. Whitens by the positive-definite marginal blocks, then uses Dykstra alternating
   projections onto the PSD cone and the affine set of identity diagonal blocks.
   This minimizes Frobenius change in **marginal-whitened coordinates**.
   Iteration stops at relative change and minimum-eigenvalue tolerance `1e-8`;
   failure to converge within 20000 iterations raises an error. Severely
   indefinite full-probe candidates can require more than 2000 iterations;
   the higher cap retains the same projection, tolerance, and explicit failure.
3. Shrinks toward block independence, preserving all marginal blocks:
   `C_final = (1-lambda) C_projected + lambda blockdiag(C_1, ..., C_m)`.
   With minimum/maximum projected, whitened eigenvalues `a,b` and `k=1000`,
   `lambda = max(0, (b-k*a)/(b-k*a+k-1))`, plus a numerical safety margin when
   shrinkage is needed. This bounds the **block-whitened condition number** to
   1000; it does not bound the raw condition number of ill-conditioned marginals.
   The choice 1000 is an explicit numerical policy, not a calibrated statistical
   optimum. There is no hidden eigenvalue floor in the group model.

Singular, nonpositive, or nonfinite marginal blocks remain unavailable (NaN
rows/columns), with a warning; the patch cannot make them positive definite while
preserving them. Unavailable cross-covariances between otherwise identifiable
signals are explicitly approximated by zero with a warning. A full-covariance
group model still rejects unavailable/numerically singular matrices.

Inspect `rec["stats"].covariance.attrs` for the method, source diagnostic scope,
unavailable blocks, missing cross entries, iteration count, shrinkage, minimum
block eigenvalues, raw and block-whitened condition numbers, and correction size.
`relative_block_correction` measures projection/shrinkage after restoring marginal
blocks; `max_marginal_block_replacement` measures a supplied block replacement
and is zero for this legacy-preserving production path. The
`marginal_source="legacy_ar_irls_joint_diagonal_blocks"` attribute distinguishes
it from the rejected individual-block version. Corrections exceeding 10% in either measure
produce a warning. Diagnostics survive pickling, condition selection, contrasts,
and ROI pooling; they describe the **source first-level covariance**, not a new
projection of a contrast/ROI matrix. Group covariance attributes retain each
subject's source diagnostics and the weighting/robust options used.

Relative to the pre-regularization legacy estimator, first-level beta estimates,
individual-coefficient SE/p/q-values, and within-signal contrasts (such as
task A minus task B for one channel/chromophore) are unchanged. Cross-signal
contrasts and ROI uncertainty can change; group estimates can also change
because their input whitening changes. This patch does not reconcile the
separate individual-fit `cov_params()`/`bse` robust-scale formulas.
**Refit first-level and then group models**; existing notebook variables and
saved results are not repaired retroactively.

### Covariance ROC and calibration validation

ROC curves, AUC, sensitivity, and channel-level Type-I error use **raw,
uncorrected channel p-values**, never BH-adjusted q-values. At a fixed threshold,
report FPR as `false positives / truly null channel tests` (and sensitivity over
truly active tests). All-null calibration uses this same channel-based FPR,
not the probability of any discovery across channels. The separate BH-FDR
experiments below are supplementary multiple-testing analyses, not substitutes
for ROC or raw channel calibration.

Run the paired Monte Carlo study with the selected project interpreter:

```bash
python -m pyBrainAnalyzIR.testing.covariance_validation \
  --replicates 100 --subjects 12 --output covariance-roc.json
```

The study writes JSON (configuration, per-replicate AUC/FPR/power/95% coverage,
paired differences, covariance diagnostics, warnings, fitting failures, and
exploratory calibration flags) and a replicate-level CSV. It compares:

- Legacy first-level joint covariance and the **removed** group eigenvalue floor,
  reproduced only in the study, never re-enabled in production.
- Repaired full-covariance weighting.
- Repaired marginal variances with explicit diagonal group weighting.

The five regimes are independent Gaussian noise, spatially correlated AR(1)
noise, correlated AR noise with channel-specific impulses/motion shifts, and an
all-null version of the latter, plus a deliberately difficult short-recording,
high-correlation/unequal-mask stress regime. These are concentration-space event GLMs
(180 seconds at 2 Hz, four channels/two chromophores, AR order search up to 8).
The stress regime uses 60 seconds, spatial correlation 0.98 rather than 0.35,
motion/outliers, and different excluded sample segments per channel to reproduce
genuinely indefinite candidates and exercise the projection/shrinkage itself.
Each group uses 12 subjects; separate mean, age, and categorical-effect models
are tested with robust fitting both on and off. Non-null scenarios include
subject random effects and heterogeneous signal noise. The first-level truth is
the actual injected subject coefficient; group truth is the population effect.
The seed, ordered regime selection, and replicate index determine the random
streams. Use the same `--regimes` ordering when reproducing a subset run.

Intervals use 2000 bootstrap resamples of **independent simulation replicates**,
averaging subjects within replicate first. Channels/chromophores are not treated
as independent Monte Carlo replications. All-zero/all-one bounded rates use
conservative exact replicate-event bounds instead of a misleading zero-width
bootstrap interval (with 100 all-zero replicates, the upper bound is 3.62%).
AUC is accompanied by null rejection
at uncorrected `p<0.05` and 95% coefficient coverage: good discrimination alone
cannot establish valid inference. Calibration flags are exploratory,
not multiplicity-adjusted confirmatory tests; absence of a flag is not proof.
The existing [`17_GroupAnalysis_ROC.ipynb`](examples/17_GroupAnalysis_ROC.ipynb)
also exercises intensity-to-concentration preprocessing, multiple conditions,
and categorical/continuous effects with ROC curves.

**PSD is a numerical property, not evidence that p-values are calibrated.**
In particular, the paired study exposed severe undercoverage in the existing
robust group covariance calculation with both legacy and repaired inputs.
The repair must not be interpreted as validating that separate estimator.
Neither group robust defaults nor the group variance estimator are silently
changed by this patch. Group fits now warn and record
`conditional_residual_collapse=True` if robust fitting produces a conditional
residual norm less than `1e-4` of the fixed-effect residual norm (in fitting
coordinates). This numerical warning identifies near-interpolation by random
effects, not all forms of miscalibration; an unflagged fit is not certified.

#### Historical results: rejected individual-block replacement

**The following recorded results evaluated the earlier, rejected patch that
replaced legacy blocks with individual-fit `cov_params()`. They are retained as
the evidence for rejecting that change, not as results for the current
legacy-block-preserving implementation. Rerunning the command above now uses
legacy marginal blocks.**

The completed validation used **500 independent datasets** (100 per regime),
12 subjects each: 6000 repaired first-level covariance matrices and 9000 group
fits across the three weighting variants, three models, and two robust settings.
There were no fitting failures. All repaired matrices were positive definite
with block-whitened condition number at most 1000. All 1200 stress-regime
candidates needed projection/shrinkage; median relative correction was 20.8%,
median shrinkage 0.00707, and maximum projection iterations 203. Thus the study
exercised actual indefinite candidates, not just already-valid matrices.

The **174 related regression tests pass**, including full marginal-block
preservation, unchanged first-level betas, missing/rank-deficient fits, unit and
permutation equivariance, contrasts, ROI covariance propagation, retained fit
data, viewer behavior, group whitening, and Monte Carlo truth/interval handling.
The three existing ROC notebook scenarios also ran at their full six-iteration
defaults; their aggregate HbO/HbR AUCs ranged from 0.919 to 0.979.

However, **inferential calibration does not pass in all regimes**. First-level
null rejection at nominal uncorrected 5% was:

| Noise regime | Legacy FPR | Repaired FPR | Repaired 95% replicate interval |
|---|---:|---:|---:|
| Independent Gaussian | 3.46% | 5.29% | 4.33-6.29% |
| Correlated AR | 3.83% | 5.25% | 4.33-6.17% |
| AR with motion/outliers | 11.63% | 14.46% | 13.29-15.71% |
| All-null AR with motion/outliers | 12.23% | 16.04% | 15.09-16.91% |
| Short, highly correlated, unequal masks | 17.25% | 22.71% | 20.83-24.58% |

Restoring individual `cov_params()` therefore worsens false-positive inflation
in the difficult settings even though it fixes covariance consistency and PSD.
In the all-null regime, first-level nominal 95% coverage was only 83.96%.
ROC AUC was broadly similar before/after; that does not rescue the p-values.

For the all-null **group mean**, repaired full weighting gave 5.38% FPR
(3.63-7.25%) without group robust fitting, versus 10.88% (7.00-15.25%) with it.
The nonrobust age model still had 8.00% FPR (6.00-10.13%): switching off robust
fitting is not a universal calibration fix. With mixed null/non-null signals,
the existing robust group mean model was much worse: under IID noise its null
FPR was 100% with legacy covariance and 99% after repair, despite AUC 1.0.
Direct inspection confirmed nearly interpolating subject random effects and
collapsed robust SEs; this is a separate, pre-existing group scale-estimation
problem, not evidence that the first-level PSD projection validated inference.

**Conclusion:** the numerical repair is verified, but it is not an empirically
validated general-purpose inference fix. The marginal AR-IRLS variance estimator
and robust group variance estimator still require further statistical work.
Do not treat successful fitting, a PSD covariance, or a favorable ROC curve as
permission to trust nominal p/q-values in these failing settings. Monte Carlo
intervals here are exploratory and apply only to the stated simulation designs.

#### Paired example-03 FDR ablation

The preceding calibration tables report **uncorrected false-positive rates**,
not FDR. For actual FDR and ROC comparisons on the example-03 intensity
simulation and preprocessing, run:

```bash
python -m pyBrainAnalyzIR.testing.covariance_fdr \
  --replicates 200 --snrs 0 1 10 --output example03-covariance-fdr.json
```

This writes JSON, per-recording CSV, and ROC/FDR plots. It fits OLS and AR-IRLS
to the same recordings at 4 Hz, keeping each fit's coefficients and degrees of
freedom fixed while comparing five covariance variants:

- `legacy_raw`: the joint covariance before the interim patch.
- `marginals_only`: replace marginal blocks with individual `cov_params()`,
  without PSD projection/shrinkage.
- `individual_blocks_psd`: the rejected repair using individual-fit marginal
  blocks, retained as an explicit diagnostic comparator.
- `legacy_blocks_psd`: project/shrink while preserving the **legacy** marginal
  blocks, isolating the numerical PSD repair from the marginal variance change.
- `current_repaired`: the current legacy-block-preserving production path;
  now identical to `legacy_blocks_psd`.

The unregularized variants are diagnostic only and may be indefinite; they are
never fed to group whitening. OLS does not use this AR-specific repair, so all
five variants must agree exactly for OLS. Individual task-coefficient p-values
must agree between `legacy_raw` and `legacy_blocks_psd`, and between
`marginals_only` and `individual_blocks_psd`, because these pairs have identical
diagonal variances. `current_repaired` now pairs with `legacy_raw`.
Off-diagonal covariance matters for multicoefficient
contrasts and downstream group weighting, not these scalar tests.

For each independent recording, the study counts false discoveries `V` and
total discoveries `R` at BH `q<0.05`; empirical FDR is the mean of
`V/max(R,1)`, **not** the fraction of null tests having raw `p<0.05`.
The primary BH family contains only HRF A across channels and both chromophores;
separate HbO/HbR families are also reported. The
`toolbox_global_q_task_subset` row instead uses the normal statistics table's
global BH correction (including drifts), then measures task-subset FDP. That
subset quantity is not global-family FDR. Under all-null simulation every
discovery is false, so FDR equals the probability of at least one discovery.
At mixed null/non-null settings, 5% is an upper control level, not a target
that a correct method must attain; the expected bound depends on the family.
Paired differences and Monte Carlo intervals use independent recordings.

The historical example-03 comparison (before switching the production path to
legacy blocks) used **200 independent recordings per
SNR** (600 recordings and 1200 fits in total). For AR-IRLS with the task-wide
BH family, the results were:

| SNR | Legacy FDR | Current repair FDR | Legacy raw-p FPR | Current raw-p FPR |
|---|---:|---:|---:|---:|
| 0 (all null) | 5.50% | 18.50% | 6.19% | 8.22% |
| 1 | 3.68% | 5.49% | 7.16% | 9.53% |
| 10 | 3.70% | 5.06% | 6.69% | 8.84% |

The paired all-null FDR increase was **13.0 percentage points**, with a
95% replicate-bootstrap interval of 8.5-18.0 points. Current all-null FDR had
an interval of 13.0-24.0%, versus 2.5-8.5% for legacy. For SNR 1 and 10,
paired FDR increases were 1.81 points (1.38-2.27) and 1.36 points (0.95-1.77).
AUC was approximately 1 for both non-null SNRs, illustrating again why ROC
discrimination alone misses this calibration regression.

The ablation identifies **marginal-block replacement**, not the off-diagonal
PSD projection/shrinkage, as the cause of the scalar-test changes:
`legacy_blocks_psd` exactly matched legacy scalar-test results on every
recording, and `marginals_only` exactly matched `current_repaired`.
The median task variance ratio (current/legacy) was 0.851. All 600 legacy
joint matrices happened to be positive definite in this experiment; after
replacing their marginal blocks, 128 candidates became indefinite and 143
needed conditioning shrinkage. Thus replacing blocks while retaining the
legacy cross-block approximation can itself introduce incompatibility.

With the normal statistics table's global q-values, all-null *task-subset*
FDP increased from 1.5% to 8.0%; this has a different correction family and
must not be mislabeled as global-family FDR. OLS was exactly unchanged across
all four variants. Its severe miscalibration on this AR-noise simulator
predates, and is unrelated to, the AR covariance repair.

This comparison motivated the current production choice: **preserve legacy
marginal blocks** while repairing joint structure. The smaller individual-fit
`cov_params()` blocks were not justified by PSD correctness alone and were
implicated in the measured FDR regression. Historical saved result labels
`current_repaired` refer to that earlier individual-block version; consult
the saved configuration/version and do not reinterpret historical numbers as
measurements of the new production estimator.

#### Legacy-block-preserving ROI and group validation

```bash
python -m pyBrainAnalyzIR.testing.downstream_covariance \
  --replicates 20 --subjects 12 --output roi-group-covariance.json
```

This staged paired study exercises the actual production ROI pooling and group
modules, not a surrogate covariance calculation:

- Physical channels; ROI pooling of channel statistics using fixed membership
  or inverse coefficient-variance weights; and ROI time-course pooling before
  GLM using fixed membership or inverse temporal-variance weights.
- First-level task A, task B, and within-signal A-B contrasts; group mean, age,
  and categorical effects with robust fitting on/off.
- Full and diagonal group weighting for both legacy and repaired covariance.
  The former legacy eigenvalue floor is reproduced only for the legacy-full
  comparator. It is not reintroduced into production group whitening.
- ROI pooling after channel group fitting, using fixed and inverse-variance
  weights, in addition to group fitting of the subject-level ROI statistics.
- IID, correlated AR, AR/motion, all-null AR/motion, and short/highly
  correlated/unequally masked recordings. The group experiment targets task A
  from simultaneous two-condition first-level fits; the ROC notebook separately
  exercises simultaneous two-condition group models.

The two disjoint ROIs combine physical channels with membership weights 1:2;
members share their population effects, so population ROI truth is defined
without estimating it from fitted data. Subject-level truth uses the actual
pooling matrix applied to injected coefficients. For time-course ROI fitting,
masks are intersected within each region so a single true pooled coefficient
exists; unequal channel masks remain in the statistics-pooling experiment.
Time-course pooling and coefficient pooling are different estimators and are
not expected to agree for AR-IRLS or data-dependent weights.

Outputs include actual BH FDR (`mean V/max(R,1)`), raw-p FPR, AUC, coefficient
coverage, bias/RMSE, independent-dataset bootstrap intervals, paired changes,
source covariance diagnostics, and explicit failures. Each BH family is the
selected task, contrast, or group term across channels/ROIs and chromophores;
drifts are not included. Subject FDPs are averaged within dataset before
bootstrapping, rather than counting correlated subjects/channels as independent
replicates. Complete-pair counts are reported for downstream fitting failures.
With only 20 datasets per regime, uncertainty in group null rates is substantial;
the run is a breadth/stress evaluation, not certification at a precise 5% bound.

For first-level ROI discrimination, run the companion experiment:

```bash
python -m pyBrainAnalyzIR.testing.downstream_covariance \
  --first-level-only --replicates 100 --output first-level-roi-covariance.json
```

This removes random subject effects, producing both exact-null and active ROIs
in non-null recordings. In contrast, the group experiment includes random
subject effects in a population-null ROI: those individual subject coefficients
are genuinely nonzero, so first-level AUC may be undefined even though group
ROC is well defined. The companion experiment uses 100 independent recordings
per regime and the same production pooling, task/contrast inference, paired
comparison, and explicit failure reporting. It does not change the group study
or the production estimator.

##### Completed legacy-block-preserving comparison

The completed paired runs used 100 group datasets (20 per regime, 12 subjects),
plus 500 independent first-level exact-null/active recordings (100 per regime):
**5,100 actual AR-IRLS fits and 12,000 group fits**, with zero recorded failures.
All seven group/ROI surfaces, three group models, robust on/off, and full/diagonal
weighting were evaluated. The existing group ROC notebook also completed its
full six iterations per scenario with the revised producer.

Numerically, 823/5,100 original joint covariance candidates were indefinite.
All repaired matrices were positive definite; 824 required normalized shrinkage,
whose maximum was 0.00791. The maximum block-normalized condition number was
1,000, no marginal blocks were unavailable, and the largest normalized
cross-block correction was 60.7%. A large projection can therefore be necessary
even when the subsequent shrinkage coefficient is small.

The required invariants held exactly:

- Physical-channel and time-course-ROI first-level betas, scalar tables, and
  within-signal A-B inference were unchanged.
- For those surfaces, legacy/repaired **diagonal-weighted group inference**
  was also unchanged across every recorded metric.
- Statistics-pooled ROI betas and pooling weights were unchanged, but their
  covariance can change because it depends on cross-signal entries.

In the separate first-level experiment, the largest absolute paired empirical
FDR change was **+1.0 percentage point**, for inverse-variance statistics pooling
in the short/high-correlation stress regime (95% paired interval 0.0-3.0 points).
The largest absolute paired AUC change was 0.005. Neither result establishes
equivalence or guarantees nominal calibration.

In particular, the all-null AR/motion experiment still showed the following
task-A BH FDR, measured over 100 independent recordings:

| First-level surface | Legacy FDR | Repaired FDR | Repaired 95% interval |
|---|---:|---:|---:|
| Physical channels | 29% | 29% | 21-38% |
| Statistics ROI, fixed membership | 17% | 17% | 10-25% |
| Statistics ROI, inverse variance | 22% | 22% | 14-30% |
| Time-course ROI, fixed membership | 18% | 18% | 11-26% |
| Time-course ROI, inverse variance | 19% | 19% | 12-28% |

These are synthetic AR/motion stress results, not the example-03 intensity
simulation. Their elevated FDR is already present with legacy covariance:
preserving the marginal blocks avoids the earlier repair-induced regression,
but does **not** solve first-level calibration under outliers.

For nonrobust full-weighted group models, the largest absolute paired FDR
change was -5.0 percentage points (95% interval -15.0 to 0.0), for the age model
after inverse-variance time-course pooling in the short/high-correlation regime.
The largest absolute paired AUC change was -0.05 (95% interval -0.125 to
0.0125), for the categorical model on the same surface/regime. These comparisons
use the explicitly study-only legacy eigenvalue floor, not a restored
production fallback. Stress-regime repaired nonrobust AUC ranged from
0.950-0.988 for mean effects, 0.900-0.950 for age, and 0.825-0.963 for
categorical effects across the seven surfaces. All-null nonrobust group-mean
FDR was 10-15%, with wide intervals at only 20 independent datasets.

The existing robust group variance problem remains unacceptable: stress-regime
age FDR was **41.7-50.0%**, and nominal 95% coefficient coverage was only
**6.3-16.3%** across the seven repaired-full surfaces. The conditional-residual
collapse diagnostic triggered in 25.9% of repaired-full robust fits across
regimes/models/surfaces, versus 43.7% of the legacy-full comparator. Numerical
repair is not a remedy for this estimator problem; an explicitly nonrobust
comparison remains necessary.

**Conclusion:** the legacy-block-preserving patch meets the numerical and
inference-preservation contracts and avoids the rejected marginal-replacement
regression. These tests do not justify calling the entire first-/second-level
inference pipeline calibrated at 5%, or treating robust group inference as
validated. The remaining marginal and robust-group variance issues require
separate estimator work; no defaults were silently changed to make the tests
appear calibrated.

#### Diagnosing elevated raw channel false-positive rates

The matched diagnostic study isolates AR-order limits, covariance formulas,
degrees of freedom, and motion components without changing production:

```bash
python -m pyBrainAnalyzIR.testing.raw_null_calibration \
  --replicates 400 --orders 1 8 30 --output raw-null.json
python -m pyBrainAnalyzIR.testing.raw_null_calibration \
  --replicates 400 --filter-transfer-only --output clean-filter-transfer.json
python -m pyBrainAnalyzIR.testing.raw_null_calibration \
  --replicates 100 --matlab-reference --output matlab-raw-null.json
```

The concentration-space simulation has the same task/noise laws as the previous
study: 360 samples at 2 Hz, two task regressors plus an intercept, eight
channel/chromophore signals, background AR(1) coefficient 0.65, spatial
correlation 0.35, five additive impulses and one eight-sample shift per signal.
All task effects are zero. Each recording shares identical background noise
across the no-motion, impulse-only, shift-only, and combined conditions.
Intervals resample 400 independent recordings, not 3,200 correlated signal tests.

Using legacy marginal covariance and raw two-sided `p<0.05` for task A:

| Motion condition | Estimated AR, limit 8 | Known background AR(1) filter | Filter estimated on clean background |
|---|---:|---:|---:|
| None | 4.22% | 3.47% | 4.22% |
| Impulses only | 13.34% | 3.41% | 3.84% |
| Short shifts only | 3.38% | 3.88% | 4.63% |
| Impulses and shifts | 11.59% | 3.84% | 4.25% |

For combined motion, the current 11.59% rate had a 95% interval of 10.28-12.81%.
Transferring the clean-background filter to the **unchanged contaminated data**
reduced FPR by 7.34 percentage points (paired 95% interval -8.31 to -6.41).
Using the known background filter reduced it by 7.75 points (-8.78 to -6.75).
Both controls are diagnostic oracles: real recordings do not supply clean
background data or known AR coefficients.

The dominant mechanism in this experiment is the mismatch between
**ordinary-least-squares AR estimation on contaminated residuals** and the
subsequent robust regression. The AR step does not use the regression's
outlier weights. Impulses attenuate the estimated background correlation:
with the AR limit fixed at one, the average sum of fitted AR coefficients was
0.634 without motion but 0.393 with combined motion (true background value
0.65). At limit eight, ordinary whitened residual lag-one correlation averaged
-0.009, deceptively near zero, while the robust task-score lag-one correlation
was **0.258**. After transferring the clean-estimated filter it was **0.004**.
The relevant robust estimating equations retain temporal dependence even
when the outlier-dominated residual autocorrelation looks white.

The other ablations do not explain away this problem:

- **AR ceiling:** raising the maximum from 8 to 30 changed combined-motion FPR
  from 11.59% to 11.56%. Limit-eight fits selected order 1.90 on average and
  hit the ceiling in only 0.5% of fits. A higher allowed order is not the fix.
- **Degrees of freedom:** production uses `n-p=357`; the separate stored
  Satterthwaite estimate averaged 335.4. Using that estimate or subtracting the
  selected AR order left the observed FPR at 11.59%. Even forcing df=30 left
  FPR at 10.47%; this is not primarily a t-distribution df problem.
- **Marginal uncertainty:** empirical beta sampling variance was 1.36 times
  mean reported variance under combined motion. Replacing legacy covariance
  with individual-fit `cov_params()` increased FPR to 15.03%. The cross-signal
  PSD repair cannot affect these preserved marginal tests.
- **Generic sandwich replacements:** the exploratory conditional HC0/HAC
  estimates did not consistently calibrate clean and contaminated conditions.
  They ignore estimated-filter/scale uncertainty and are not production fixes.

The local MATLAB NIRS-toolbox was also executed on **100 identical saved
recordings per motion condition**, using `nirs.math.ar_irls` at limit eight.
Python and MATLAB selected identical AR orders in all 3,200 signal fits;
maximum beta discrepancy was below `1.7e-15` and relative marginal-variance
discrepancy below `6e-14`. Both returned raw FPR of 4.00%, 12.88%, 3.38%, and
11.00% for no motion, impulses, shifts, and combined motion respectively.
MATLAB's Satterthwaite df and its module's final SVD covariance transformation
did not change those rejection rates. This was an explicit-design math
comparison plus reproduction of the final module covariance transformation,
not an end-to-end MATLAB preprocessing run. The internal `stats.pval` uses
the robust-fit covariance rather than the subsequently assembled `stats.covb`;
those internal p-values must not be confused with ChannelStats inference.

Thus the tested inflation is reproduced by MATLAB, not evidence of a Python
porting discrepancy. The next estimator work should target **motion-robust AR
estimation/whitening together with marginal uncertainty**, followed by matched
clean/motion calibration and downstream ROI/group validation. No production
formula, df policy, or default was changed by these diagnostic experiments.
Correct marginal p-values are necessary for BH control; the dependence
assumptions of BH must also hold.

#### Robust AR coefficient estimation

The default `ar_estimator="robust"` now wraps the forward/backward AR regression
in a robust fit at every AR-IRLS outer iteration:

1. Retain the existing **OLS BIC** choice of AR order, including order zero.
2. Exclude regression rows whose lagged residual predictors exceed 4.685 robust
   scales from the residual series' median (scale = median absolute deviation
   divided by 0.6745). These are excluded **only from AR coefficient estimation**,
   not deleted from the time series. This protects against motion spikes in
   predictors, which ordinary residual-only reweighting can miss.
3. Refit the selected AR regression using the existing leverage-adjusted
   bisquare IRLS implementation. Use those coefficients for whitening and then
   the existing robust GLM fit. The AR regression allows up to **500 iterations**
   (rather than 50): correlated lag predictors and near-zero coefficients can
   converge slowly under the unchanged coefficient-wise relative tolerance.
   The task-response regression retains its legacy 50-iteration budget.

The outer convergence rule, marginal covariance formulas, degrees of freedom,
and joint-covariance repair are unchanged. Bisquare iteration-limit warnings
are surfaced rather than silently suppressed. This is robust coefficient
estimation, **not a robust BIC order-selection criterion**.

For pipeline users:

```python
job = GLM()
job.options["ar_estimator"] = "robust"  # default
job.options["ar_order"] = 8
```

The low-level solver accepts `fit(..., noise_model="ar_irls",
ar_estimator="robust", ar_order=8)`. Set `ar_estimator="ols"` to reproduce the
original MATLAB-compatible AR coefficient estimator. MATLAB comparison helpers
explicitly select `"ols"`; their reports identify this choice. The GLM pipeline
now forwards its `ar_order` option to the solver as intended (previously that
option was omitted, leaving the solver's default of 30).

Per-channel fit results retain the estimator name, AR coefficients and whitening
filter. Statistics retain the estimator names in
`stats.covariance.attrs["ar_estimator"]`; this metadata survives serialization,
contrasts and ROI pooling. Refit first-level and downstream models to apply the
new estimator; saved fits are not retroactively changed.

##### Matched raw-channel validation

The following all-null comparison uses **raw task-A p-values**, not BH q-values,
400 independent recordings per condition, and the same saved seed/noise/design
as the original AR diagnostic. The AR order limit is eight:

| Motion condition | OLS AR raw FPR | Robust AR raw FPR |
|---|---:|---:|
| None | 4.22% | 4.09% |
| Impulses only | 13.34% | 4.28% |
| Short shifts only | 3.38% | 3.94% |
| Impulses and shifts | 11.59% | 3.84% |

Combined-motion FPR fell by **7.75 percentage points**, with a paired
95% recording-bootstrap interval of -8.72 to -6.75 points. Mean lag-one robust
task-score correlation fell from 0.258 to 0.012. Limits of one and 30 were also
tested; complete results are retained in the study output.

An additional 200 matched recordings per condition injected task A=0.3 and
task B=0.15 into four of eight signals, leaving four exact-null signals.
With combined motion, raw null-channel FPR fell from **9.00% to 4.38%**.
Raw sensitivity changed from **98.63% to 96.50%** (paired difference -2.13
points, 95% interval -3.38 to -1.13). Clean-data sensitivity remained 98.13%
for both estimators. Better calibration should not be confused with preserving
the artificially high sensitivity of an anticonservative test.

Reproduce the robust runs with:

```bash
python -m pyBrainAnalyzIR.testing.raw_null_calibration \
  --ar-estimator robust --replicates 400 --orders 1 8 30 --output robust-ar-null.json
python -m pyBrainAnalyzIR.testing.raw_null_calibration \
  --ar-estimator robust --active-effects --replicates 200 --orders 8 \
  --output robust-ar-active.json
```

Use `--ar-estimator ols` for the matched original-estimator controls. Raw FPR
and raw sensitivity are scored separately; intervals resample independent
recordings, not correlated channels. Regression tests additionally cover
AR(1)/AR(2) coefficient recovery with contaminated targets and lag predictors,
scale equivariance, masked samples, threaded fitting, pipeline option forwarding,
positive covariance, metadata, and the legacy mode.

These results remove the observed inflation in the tested AR(1)/additive-motion
simulation; several conditions remain mildly conservative. They do not certify
arbitrary AR processes, motion distributions, or real recordings. The legacy
channel-specific robust group variance collapse is separate; the default group
structure now uses a shared subject intercept as described above. Earlier
large ROI/group study numbers describe the **OLS-AR** estimator, not a new
calibration certification of robust-AR group inference.

##### Robust-AR channel, ROI, group, and joint-covariance validation

The downstream study now separately reports **raw sensitivity** (`sensitivity`)
and raw null-test rejection (`fpr`); its historical `power` and `fdr` fields
remain supplementary BH metrics, not ROC calibration measures. Cross-signal
half-sums and half-differences additionally test the full covariance for task A,
task B, and A-B, including both between-channel and between-chromophore terms.
Their empirical error variance is computed across independent recordings for
each fixed contrast and compared with mean estimated contrast variance.
Intervals resample recordings, not individual contrast tests.

With robust AR, 100 independent exact-null/active recordings per regime gave
the following raw task-A false-positive rates:

| Noise regime | Channel | Statistics ROI, fixed | Statistics ROI, IV | Time-course ROI, fixed | Time-course ROI, IV |
|---|---:|---:|---:|---:|---:|
| IID | 3.50% | 3.00% | 3.00% | 3.00% | 2.00% |
| AR(1)=0.65 | 3.50% | 3.50% | 3.00% | 2.50% | 3.00% |
| AR + motion, some active signals | 4.75% | 3.50% | 3.00% | 5.00% | 5.50% |
| AR + motion, all null | 3.50% | 4.75% | 4.75% | 4.50% | 4.00% |
| Short AR(1)=0.98, unequal masks | 12.75% | 10.00% | 15.00% | 13.50% | 12.00% |

These are repaired-covariance results. The exact-null experiment has no
subject random effects; the separate group experiment does. Channel raw
sensitivity under AR + motion was 79.0%; the four ROI paths ranged from
81.0% to 84.5%. These amplitudes/designs differ from the earlier matched
motion diagnostic, so its sensitivity numbers are not directly comparable.
On the same historical experiment, changing OLS AR to robust AR reduced
active-motion channel FPR from 10.50% to 4.75% and all-null-motion channel FPR
from 11.13% to 3.50%. Active-motion ROI FPRs fell from 11.0-15.5% to
3.0-5.5%. Stress-channel FPR improved from 18.50% to 12.75%, but remains
clearly above nominal.

Cross-signal task-A contrasts from the channel fits had 95% interval coverage
of 96.41%, 96.72%, 95.69%, and 96.28% in the IID, ordinary AR, active-motion,
and all-null-motion regimes respectively. Empirical/estimated variance ratios
were 0.84, 0.85, 0.94, and 0.85. **The short, highly correlated stress regime
still fails:** coverage was 82.19%, raw null-contrast FPR 15.25%, and the
variance ratio 1.79. Its marginal coverage was also low (83.13%). Thus this
remaining failure is not exclusively an off-diagonal covariance problem,
and robust AR is not a universal covariance-calibration fix.

Of the 1,500 actual channel/time-course ROI GLMs in this experiment, 251 raw
joint matrices remained indefinite. All repaired matrices were positive
definite with normalized condition at most 1000 and no unavailable marginal
blocks. There were 164 recording/surface failures of **unrepaired** cross-signal
contrast batches because of nonpositive variances; these are explicitly
counted, not floored or assigned passing rates. No repaired contrast batch or
scalar test failed. Comparisons conditional on successful unrepaired batches
must not be interpreted as unbiased paired calibration comparisons.

Keep the marginal-block-preserving joint repair: robust AR does **not** make
the original cross-covariance assembly inherently PSD. Repair still leaves
channel and time-course ROI marginal blocks and within-signal contrasts
exactly unchanged for the same fit. Statistics-ROI covariance and full group
weighting can change because they use cross-signal entries.

The following historical group extension used the old **channel-specific**
random effects, not the current shared-subject default. It ran 20 independent
datasets per regime, twelve subjects
each, with mean/age/categorical formulas, robust group regression on/off,
full/diagonal weighting, all five channel/ROI input surfaces, and two
group-then-ROI pooling paths. Together with the exact-null experiment this
comprised **5,100 actual AR-IRLS GLMs and 12,000 group fits**. All repaired
inference paths completed; the only recorded failures were the 164 invalid
unrepaired joint-contrast batches above. Across these GLMs, 859 unrepaired
joint matrices were indefinite; all repaired matrices met the numerical
condition target. Maximum projection iterations were 183, so increasing the
full-probe iteration cap did not change these simulation results.

**Group inference is not yet validated as nominal.** With nonrobust group
regression, channel mean/age/categorical raw FPRs were 0-1.25% in the ordinary
active-data regimes, with 97.5-100% sensitivity. However, the all-null-motion
group mean had channel FPR 10.0% (95% dataset interval 2.5-18.1%); across
channel/ROI paths it ranged from 10.0% to 18.75%. Fixed statistics-ROI FPR was
17.5% (interval 7.5-30.0%). These point estimates and broad twenty-dataset
intervals do not justify certifying a 5% rate.

The separate **robust group regression** problem is more severe and was not
silently changed: channel age raw FPR was 100% in IID and short-AR stress,
68.75% in ordinary AR, and 40.0% with active motion. Corresponding IID and
stress rates were already 100% and 92.5% with OLS AR, so this is not created by
making first-level AR estimation robust. In active motion robust AR improved
that group's FPR from 75.0% to 40.0%, still unacceptable. This agrees with the
previously identified conditional-residual variance-collapse limitation;
positive-definite first-level inputs do not repair that second-level estimator.
Its collapse warning uses a severe residual-ratio threshold and does not
detect every poorly calibrated case. Keep first-level robust AR distinct from
the group's `robust` option. A group uncertainty/df investigation remains
necessary; changing group defaults or covariance formulas to obtain favorable
ROC curves was deliberately avoided.

Both ROC notebooks were also executed end-to-end without replacing their saved
outputs. Example 03 ran its full ten iterations each for AR-IRLS, OLS, and
group inference. AR-IRLS channel raw FPR was 5.0%/2.5% (HbO/HbR), versus
68.75%/68.75% for OLS on serially correlated data. All had AUC 1.0, illustrating
why discrimination alone cannot establish calibration. Example 17 ran its
full six independent datasets per mean/age/categorical scenario with twelve
subjects and simultaneous A/B tasks, using robust first-level AR and the
notebook's explicit nonrobust group fit. Across chromophores, raw FPR ranged
5.56-7.64% for the mean scenario (including null age terms), 1.04-4.17% for age,
and 5.21-9.38% for categorical effects. Six datasets are functional validation,
not precise certification of a 5% rate.

Reproduce the robust-AR downstream validation with:

```bash
python -m pyBrainAnalyzIR.testing.downstream_covariance \
  --replicates 20 --subjects 12 --regimes iid correlated_ar ar_outliers \
  --output robust-roi-group-main.json
python -m pyBrainAnalyzIR.testing.downstream_covariance \
  --replicates 20 --subjects 12 --regimes ar_global_null ar_stress \
  --output robust-roi-group-null-stress.json
python -m pyBrainAnalyzIR.testing.downstream_covariance \
  --first-level-only --replicates 100 --output robust-first-joint.json
```

In these outputs `legacy` denotes **unrepaired covariance from robust AR**,
not the old OLS AR coefficient estimator. Group comparisons labeled
`legacy_full_floor` reproduce the old eigenvalue floor for the experiment
only; production does not silently introduce that precision.

## About The toolbox


## Development

Install the development/CI tooling with:

```bash
pip install -r requirements-dev.txt
```

### Running the tests

```bash
pytest
```

The test suite lives in [`tests/`](tests) and covers the option-variable data
classes, every pipeline module, the GLM and ROC analyses, and the group-level
mixed-effects models.

### Linting

```bash
flake8 pyBrainAnalyzIR tests   # style, configured in setup.cfg
pylint pyBrainAnalyzIR tests   # static analysis, configured in .pylintrc
```

All three checks run automatically on every push and pull request to `main`
via GitHub Actions (see the badges at the top of this file).

## Citation

If you use this software in your research, please cite:

> Santosa, Hendrik, Xuetong Zhai, Frank Fishburn, and Theodore Huppert. "The NIRS brain AnalyzIR toolbox." Algorithms 11, no. 5 (2018): 73

## Acknowledgements
This work is funded by the National Institutes of Health to Theodore Huppert, PhD.  R01EB028248. Brain AnalyzIR: A software platform for improving scientific rigor in functional NIRS statistical analysis


Built on top of the [cedalion](https://github.com/ibs-lab/cedalion) framework developed by the IBS Lab.
