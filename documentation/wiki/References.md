# References

1. Santosa H, Zhai X, Fishburn F, Huppert T. The NIRS Brain AnalyzIR Toolbox.
   *Algorithms*. 2018;11(5):73.
   [https://doi.org/10.3390/a11050073](https://doi.org/10.3390/a11050073)
2. Huppert T, et al. MATLAB NIRS Toolbox / AnalyzIR Toolbox source and method
   implementations.
   [https://github.com/huppertt/nirs-toolbox](https://github.com/huppertt/nirs-toolbox)
3. Cedalion fNIRS analysis framework and data model.
   [https://github.com/ibs-lab/cedalion](https://github.com/ibs-lab/cedalion)
4. GitHub Docs. Adding or editing wiki pages.
   [https://docs.github.com/articles/adding-or-editing-wiki-pages](https://docs.github.com/articles/adding-or-editing-wiki-pages)

## MATLAB method correspondence

The Python toolbox includes conceptually corresponding modules for common
AnalyzIR workflows. Consult and cite the exact MATLAB source revision used:

- Python `GLM` and AR-IRLS settings ↔ MATLAB
  [`nirs.modules.GLM`](https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmodules/GLM.m)
  and
  [`nirs.modules.AR_IRLS`](https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmodules/AR_IRLS.m).
- Python `MixedEffects` ↔ MATLAB
  [`nirs.modules.MixedEffects`](https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmodules/MixedEffects.m).
- Python `resting_state_connectivity` ↔ MATLAB connectivity/sFC methods,
  including
  [`nirs.modules.Connectivity`](https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmodules/Connectivity.m).
- Python `MixedEffectsConnectivity` ↔ MATLAB
  [`nirs.modules.MixedEffectsConnectivity`](https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmodules/MixedEffectsConnectivity.m).
- Python hyperscanning ↔ MATLAB
  [`nirs.modules.Hyperscanning`](https://github.com/huppertt/nirs-toolbox/blob/master/%2Bnirs/%2Bmodules/Hyperscanning.m).

This table states intended method-family relationships; it is not a claim that
the algorithms, defaults, degrees of freedom, preprocessing, or outputs are
identical. The current Python-vs-MATLAB comparison helper and its test
configuration are described in [Testing and MATLAB Comparison](Testing-and-MATLAB-Comparison).
