# Higher-order information and prediction in cancer co-mutation

Code and scalar results for *Separating higher-order information from prediction in cancer co-mutation*, by Freeman Hui. Release **v0.11.0** retains the benchmark implementation, the exact 24-case test suite, a portable ten-variable synthetic demonstration, and scalar records sufficient to reaggregate the S12 benchmark tables and redraw Figures 1 and 2. The older v0.9.0 and v0.10.0 tags are retained. It adds saved-null diagnostics, broader installation constraints and an attainable-rank public demo. The repository name remains stable so existing links continue to work.

## Install

Study environment: Python 3.11.5, NumPy 1.26.4, SciPy 1.13.1, pytest 8.3.3; CPU and one native numerical thread. `requirements-study.txt` preserves exact numerical/plotting versions. Installation accepts Python >=3.11, NumPy 1.26.4 through the 2.x family and SciPy 1.13.1 through 1.x. These constraints allow compatible releases; they do not claim every version combination has been tested. See `VALIDATION.md` for author-tested versus referee-reported environments. From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,figures]'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

## Reproduce the reported figures and tables without fits

```sh
python -m beyond_pairs.route12_benchmarks.reproduce --data results --output review-output --figures
```

This reaggregates every released evaluation decision using the released calibration statistics, checks the stored rank numerators, recomputes conditional Clopper–Pearson intervals, paired differences and pool-level uncertainty, and writes `tables.json`, `figure1-prediction.pdf/png` and `figure2-lrt-mask.pdf/png`. It includes the complete matched-mask and sensitivity tables, all 48 coordinate summaries, the full 120-row rounded coefficient table, sandwich diagnostics and posthoc rank-reference tails. This is result replay, not regeneration of the underlying biological fits or independent external replication.

## Compare saved null calibration and evaluation statistics

```sh
python -m beyond_pairs.route12_benchmarks.null_diagnostic --data results --output null-diagnostic.json
```

This no-fit posthoc comparison reports the one-sided Mann–Whitney statistic (evaluation greater), two-sided asymptotic KS test, mean shift, independent-cohort SE, and SE across 20 pool mean differences. It independently checks U by pair counts and KS distance by empirical CDFs. Reported p-values are exploratory and unadjusted; equality tests cannot prove exchangeability or identify a procedural cause. The optional Binomial(20, .5) sign-tail reference assumes equally likely signs, which equal distributions and unequal sample counts do not ensure.

## Run a complete public synthetic pipeline

```sh
python -m beyond_pairs.route12_benchmarks.portable demo --output surrogate-output
```

The deterministic positive ten-variable law is uniform Q and P(x)=Q(x)[1+0.3 product of the first three ±1 spins]. All single/pair moments are exactly shared. The command registers its roster before drawing, creates 278 artificial cohorts in two pools (99 null calibration, 20 null evaluation and 20 alternative cohorts each), performs 1,946 pair/triple fits, retains endpoints, runs the LRT and fully learned residual test, the nine cutoff/mask settings and ten pooled mask/map/weight combinations, and produces calibrated decisions and exact surrogate sandwich traces. It uses no repository-private paths or data. The measured default run completed in 84.473 seconds on the study machine (hardware-dependent). The default B=99 calibration pool has minimum add-one p=0.01, so rejection at alpha=.05 is attainable. The small evaluation roster tests the pipeline and **does not establish calibration, estimate reliable power or reproduce the MSK study**. The seed namespace is new in v0.11.0; prior demo records remain unchanged.

Larger runs require an explicit `--maximum-cohorts` and remain the caller's resource decision. For the published repeated-pool roster on a separately obtained law, use `--N 1557 --pools 20 --calibration 999 --null 500 --alternative 200 --maximum-cohorts 33980 --laws allowed-laws.npz`; repeat at N=6274 if resources permit. The portable driver computes additional LRT/sensitivity/factorial diagnostics on those cohorts, unlike the original repeated-pool study, which evaluated only residual D. Its new seed namespace is deliberately distinct from the original study and it must not be described as an identical rerun. Original producer code/seed schemes are in `provenance/v11-producers/`.

## Explicit inputs and generic routines

`portable cohort --counts five-fold-counts.npy --output cohort-output` accepts a 5 × 2^d integer count array for d=3..12 and fits/scores D and the pooled LRT without a known evaluation law. This counts-only command performs no calibration; external input provenance is unknown. State order is the canonical least-significant-bit-first enumeration in `designs(d)`. `--laws` for a synthetic demonstration requires an NPZ with positive normalized same-order `q` and `p`; q must be a pairwise exponential law. Canonical q/p hashes, dimension and the input-file digest are registered before draws; custom-law provenance remains the caller's responsibility. NumPy pickle loading is disabled. Numerical failures retain endpoints and terminate, rather than silently remove cohorts or relax tolerances.

The unchanged `core.py` exposes `pooled_lrt`, `fully_learned`, `oracle_factorial`, `sandwich`, `rank_decisions`, and `pool_summary`. `route12.representation.traced_layers` and `traced_mixture` accept positive finite laws/mixtures for signed decompositions. They perform fits; inspect costs before use. Natural-coefficient recovery is a held-fold-size weighted average followed by SD over independent cohorts, as documented in the original producer. Restricted hidden coordinates or biological law files are not supplied.

## Tests, provenance and numerical limits

```sh
python -m pytest -q tests/route12_benchmarks/test_known_v11.py
python -m pytest -q tests/route12_benchmarks/test_portable_v12.py
```

The first command is the exact 24-case suite used for the scientific computation. Its original completed result is inherited in `provenance/known-v11.json`; release validation does not rerun that consumed formal research phase. The second suite tests the new public-input boundaries. Source hashes, validation scope, retained failed development attempts and public surrogate evidence are documented in the manifests and `VALIDATION.md`. No cross-platform numerical guarantee is claimed. Near-singular learned weights can amplify equivalent floating-point representations; the full-mask OR/LW comparison remains flagged in the released results.

Calibration is under a specified known null law, not a cohort-refitted parametric bootstrap or composite-null guarantee. Retrospective sensitivity does not replace the original prespecified 1e-10/5 residual setting. Original private prospective records are not a retrospective public preregistration.

## Permissions and exclusions

MIT applies to the original source, tests, examples and documentation in `SOURCE-MANIFEST.json`; dependency notices and the legacy notice are retained. Patient data, patient-derived count tables, fitted biological probability laws/weights, unrounded parameter archives, credentials and private Git history are excluded. Provider terms remain separate: the author has not established how MSK-CHORD's CC BY-NC-ND 4.0 terms apply to derived-model redistribution. Obtain permitted model/data access independently. Code permission does not clear those assets. The rounded coefficient table is the already reported numerical summary, not an executable model release.
