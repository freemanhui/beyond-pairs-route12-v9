# Route12 v9: target-specific co-mutation diagnostics

Public code release v0.9.0, 16 October 2026. This source package contains the
approved portable numerical method, deterministic toy and reconstruction
instructions. It includes no patient data, cohort identifiers, fitted biological
model, restricted fitted-law samples or credentials. Publishing this code does
not constitute a full scientific refit. The source and dependency notices are
identified in SOURCE-MANIFEST.json and RELEASE-MANIFEST.json.

## Environment and first example

Use Python 3.11.5, NumPy 1.26.4 and SciPy 1.13.1. Dependency binaries are not
bundled. Obtain/install them separately under their notices; installation or
network access is not performed by the example. Native numerical threads:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHONPATH=src python3.11 -B examples/demo.py
```

The deterministic three-spin parity example enumerates eight states and five
identical 40-observation fold-count tables. The exact pairwise parameters are
zero, supplied analytically. It performs **zero fits and zero random draws**.
Each raw U statistic is 216/1560, its conditional quadratic target is 0.16,
and its oracle information is 1-H_binary(0.7), approximately 0.118709 bits.
The quadratic and entropy targets differ. An independent implementation checks
all five positions. This is a software/arithmetic installation check, not a
new cancer experiment or full scientific replication.

## Saved-result reconstruction

For permitted user-supplied finite state counts and fitted endpoints:

```sh
PYTHONPATH=src python3.11 -B examples/replay_counts.py own-input.npz
```

The NPZ must contain `tables` (5 by 2**p, nonnegative integer counts), strictly
positive normalized `p` (known evaluation law), `q0` (known null reference),
and `parameters` (5 by p*(p+1)/2, finite pairwise training parameters). Arrays
use the binary-state index sum(2**i*x_i), singles then lexicographic pairs,
and triples in lexicographic order. Pickle loading is disabled. Each parameter
vector must fit the other four folds, within moment1e-9, entropy-dual1e-7 bits
and normalization1e-12. The program refuses failed endpoints, material negative
spectra and incomplete shape contracts; it never silently fits missing inputs.

`replay.py` reconstructs A oracle, B estimated residual/fixed weight and mask,
C estimated weight/fixed mask, and D estimated weight/mask. The spectral cutoff
is 1e-10 of the largest positive eigenvalue; a learned mask requires training
expected triple count >=5. `verification.py` independently reconstructs the
same maps and contractions. Outputs distinguish raw U and its conditional
quadratic target, and multiply each by 1000/(2*ln(2)) only for the displayed
local information scale. Neither is a qualified unbiased estimator of L3.

`information.py` separates unfloored oracle score, KL and finite-sample
variance/range expressions. Its numeric reference label does not enclose an
exact machine-certified reference. `calibration.py` implements conservative
add-one ranks, fixed planned-denominator exact binomial summaries and the
continuous-pool tail parameters. `variance.py` gives the same-fixed-kernel
split/pooled identity. These targets and assumptions must remain separate.

## Scientific reproduction and its access boundary

The cancer-study frozen laws and patient data are excluded. Author permission
is recorded separately from providers' conditions. MSK-CHORD is under
CC BY-NC-ND4.0; applicability to derived fitted models remains unqualified.
No synthetic sample is declared cleared solely because it was generated from
a restricted fitted law. Independent researchers need lawful access and must
check the applicable provider conditions before obtaining data or models.

The accompanying private scientific archive binds all 4,000 retained cohorts,
20,000 reviewed positions (including exactly two recovered P2 fits), and the
separate 4,398 fresh cohorts/21,990 accepted P2 fits. Archive reconstruction is
not a fresh refit. It is available for permission-scoped review, not bundled here.

A full fresh refit is a separate route requiring a prospectively registered
run and appropriate law access. Use the exact known pair law and frozen
alternative, five-fold integer largest remainders of source proportions
[1250,1195,1298,1268,1263], and seed namespace
SeedSequence([20261016,9,1,N,law_id,purpose,index]). For each N1557/6274, Tier1
uses law0/purpose0/index0..998 for calibration, law0/purpose1/index0..999 for
null evaluation, and law7/purpose1/index0..199 for alternatives. The four
pilot index0 cases inherit inside this roster; do not add them again. The
predeclared Tier2 prefixes499/500/200 are only a cost-admission fallback,
not an optional second calibration experiment.

The included MIT-permitted `order_ladder.maxent` supplies the original
zero-start Newton method; `finite_sample_ladder.py` supplies historical
projection and floored score helpers. Preserve all failed endpoints and check
moment/dual/normalization gates before passing coefficients to `replay.py`.
The package imports but does not execute an optimizer in its isolated review.
These generic helpers are not a newly executed full-study pipeline: the
private owning run controllers, resource supervision, draw identities and
all retained endpoint receipts remain separate archival evidence.

Freeze D as primary before outcomes. Use calibration p=(1+#(Tcal>=Tobs))/1000
and alpha.05, with conditional null CP upper<=.08 and conservatively pilot-
charged power lower>=.80. Runtime-based tier admission does not by itself
establish an unconditional fixed-B rank theorem. Our reported null counts
52/1000 and38/1000, and alternative174/200 and200/200, concern one specified
null/frozen alternative. No patient/composite-null claim follows. All700 saved
finite-sample Bernstein expressions remain zero; no positive certificate is
claimed. Refit costs, needed access and fresh-run approval are not supplied by
successful archive or import checks.

## Rights and scope

`SOURCE-MANIFEST.json` identifies the exact admitted files. `LICENSE` applies
MIT to the new original v9 method/example/documentation files and the four
unchanged older files explicitly covered by the existing ten-file permission.
`LEGACY-LICENCE.txt` retains that scoped historical grant; it is not a blanket
licence for any other repository file. External NumPy/SciPy notices are included
verbatim in `notices/`. Their bundled third-party conditions remain separate.
Do not infer model/data permission from a code licence. This candidate excludes
private execution I/O/controllers, dynamic loaders and public/network actions.
