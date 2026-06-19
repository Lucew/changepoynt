# Guide to Subspace-Based Algorithms

This guide covers the subspace-based algorithms in `changepoynt`:

- `SST`: Singular Spectrum Transformation for one-dimensional time series.
- `ESST`: Entangled Singular Spectrum Transformation for one-dimensional time series.
- `MSST`: Multivariate Singular Spectrum Transformation.
- `MESST`: Multivariate Entangled Singular Spectrum Transformation.

These methods build Hankel or block-Hankel trajectory matrices from local subsequences, extract low-rank structure, and turn a difference between past and future structure into a change score. They are most useful when a change affects temporal shape, periodicity, or correlations between channels rather than only the mean or variance.

This is a configuration guide, not a claim that one detector is best for every signal. The published ESST evaluation concerns planetary drilling data, while SST and IKA-SST were introduced in different application settings. Treat the recommendations below as starting points and validate them on representative data.

For a parameter-by-parameter workflow with a stronger runtime focus, see [Tuning Subspace Methods](guides/tuning-subspace-methods.md).

## Quick Recommendation

For a first run on a one-dimensional signal, compare `SST` and `ESST` with the same window scale:

```python
from changepoynt.algorithms.esst import ESST
from changepoynt.algorithms.sst import SST

window_length = 120  # for example, one expected period at the current sample rate
sst_score = SST(window_length, method="rsvd").transform(signal)
esst_score = ESST(window_length, method="rsvd").transform(signal)
```

For multivariate data with shape `(n_samples, n_channels)`, start with `MESST` when you want the ESST score or `MSST` when you want the classic SST score:

```python
from changepoynt.algorithms.messt import MESST

detector = MESST(
    window_length=period_or_change_duration,
    method="rsvd",
    use_fast_hankel=False,
)
score = detector.transform(signal)
```

Start by tuning `window_length`. Keep `rank` small, use the implementation defaults for `n_windows` and `lag` during the first sweep, and increase `scoring_step` only after checking that coarser scoring still resolves the changes of interest.

## Which Algorithm Should I Use?

| Situation | Suggested algorithm | Notes |
| --- | --- | --- |
| One-dimensional signal, entangled past/future score | `ESST` | Decomposes the combined trajectory matrices and uses several characteristics in its score. |
| One-dimensional signal, classic SST score | `SST` | Offers the widest choice of scoring and decomposition methods, including IKA. |
| Multivariate signal, classic SST score | `MSST` | Use when several channels jointly describe the operating state. |
| Multivariate signal, ESST-style score | `MESST` | Applies the entangled score to a block-Hankel representation of all channels. |
| Large trajectory matrices | A supported method with `use_fast_hankel=True` | Benchmark both representations; the crossover depends on shape, method, and hardware. |

For multivariate algorithms, pass an array with samples on axis 0 and channels on axis 1:

```python
score = MESST(window_length=100).transform(signal)  # signal.shape == (n_samples, n_channels)
```

## How the Score Is Constructed

At each evaluated position, the algorithms compare two local trajectory matrices. A univariate matrix has approximately `window_length` rows and `n_windows` columns. A multivariate block-Hankel matrix has approximately `n_channels * window_length` rows.

- `SST` and `MSST` decompose past and future representations separately and compare their subspaces.
- `ESST` and `MESST` combine the representations before decomposition and compute an entropy-style score from the resulting singular vectors.
- A high score means the local low-rank representations differ. It is evidence of a structural change, not a calibrated probability.

The returned score has the same length as the input. Initial positions that do not have enough context remain zero, and `scoring_step > 1` fills the positions around each evaluated score. Use `detector.first_score_position` to find the first potentially valid score.

## Recommended Tuning Workflow

1. Decide which changes matter and express their duration in samples.
2. Choose univariate (`SST`/`ESST`) or multivariate (`MSST`/`MESST`) input.
3. Sweep a small set of `window_length` values around the expected temporal scale.
4. Keep `rank`, `n_windows`, and `lag` at their defaults for this first comparison.
5. Compare score traces on representative normal sections and known or plausible changes.
6. Tune `rank`, then `n_windows` and `lag`, only when the score reveals a specific problem.
7. Use `estimate_runtime()` to compare viable configurations, then test `scoring_step` and fast Hankel acceleration.

Do not select parameters from the height of a single known peak. Check multiple events and normal operating regions so that the configuration does not simply memorize one example.

## Parameter Map

| Parameter | Meaning | Practical default |
| --- | --- | --- |
| `window_length` | Length of each subsequence in the trajectory matrix. This is the main temporal scale. | Set it from the characteristic duration or period of the behavior you want to compare. |
| `n_windows` | Number of neighboring subsequences used to build each trajectory matrix. | For `SST`/`MSST`, default is `window_length`. For `ESST`/`MESST`, default is `window_length // 2`. |
| `lag` | Distance between the past and future trajectory matrices. | For `SST`/`MSST`, default is `n_windows // 3`. For `ESST`/`MESST`, default is `n_windows`. |
| `rank` | Number of characteristic sequences retained from the decomposition. | Start with `5`; reduce to `2` or `3` for very simple signals, increase only if important structure is missed. |
| `method` | Decomposition/scoring method. | Use `rsvd` for ESST/MESST; use `ika` or `rsvd` for SST/MSST. |
| `scoring_step` | Number of samples between evaluated score positions. | Start with `1`; use `2`, `5`, or `10` for faster scans on long signals. |
| `use_fast_hankel` | Uses implicit/FFT-based Hankel products instead of materialized Hankel matrices where supported. | Keep `False` for small windows; try `True` once `window_length` is a few hundred samples. |
| `scale` | Min-max scales each signal or channel to `[1, 2]` before scoring. | Keep `True` when channels have incomparable numerical ranges; use `False` when amplitude relationships carry meaning or preprocessing is already defined. |
| `mitigate_offset` | Subtracts column-wise offsets for univariate `SST`/`ESST`. | Use when local level offsets should not drive a shape comparison. It cannot be combined with `use_fast_hankel`. |

## Choosing `window_length`

`window_length` should be large enough to contain the local shape you want the algorithm to compare. Convert physical duration to samples before tuning:

```python
window_length = round(duration_seconds * sampling_rate_hz)
```

Good starting points:

- For periodic signals, begin near one period and compare nearby multiples.
- For transients, begin near the duration of the shape that should be represented.
- When the scale is uncertain, try a logarithmic or otherwise coarse sweep instead of tuning one sample at a time.
- For slowly changing operating modes, use a longer window and expect more context and latency.

The trade-off is important:

- Too small: the trajectory matrix cannot represent the behavior, so the score becomes noisy or local.
- Too large: detections are smoother and delayed, and runtime grows quickly.
- Much larger than the relevant pattern: the score is smoother, more context is required, and short changes may be diluted.

The SST literature recommends setting `n_windows = window_length`, `lag` to about one third of that value, and `rank = 5`. The package uses these defaults for `SST` and `MSST`. For `ESST` and `MESST`, the package instead defaults to `n_windows = window_length // 2` and `lag = n_windows`.

| Algorithm | Default `n_windows` | Default `lag` | Default `method` |
| --- | --- | --- | --- |
| `SST` | `window_length` | `max(n_windows // 3, 1)` | `ika` |
| `ESST` | `window_length // 2` | `n_windows` | `fbrsvd` |
| `MSST` | `window_length` | `max(n_windows // 3, 1)` | `ika` |
| `MESST` | `window_length // 2` | `n_windows` | `rsvd` |

## Choosing `n_windows` and `lag`

Usually, set `window_length` first and leave `n_windows` and `lag` at their defaults.

Change `n_windows` when:

- The signal structure changes rapidly: use a smaller `n_windows` so the reference region contains less stale behavior.
- The normal behavior is variable: use a larger `n_windows` so the trajectory matrix sees more examples of normal variation.
- Memory or runtime is too high: reduce `n_windows`, because the Hankel matrix has more columns when `n_windows` is larger.

Change `lag` when:

- You need earlier detections: reduce `lag`, accepting a more local comparison.
- You want to compare more separated behavior: increase `lag`, accepting more delay.
- Your change unfolds slowly: choose a lag comparable to the transition duration.

Larger `n_windows`, `window_length`, or `lag` also increases the required context. Check it directly rather than estimating it by eye:

```python
total_region, matrix_region = detector.covered_regions()
print(total_region, detector.first_score_position)
```

## Choosing `rank`

`rank` controls how many characteristic subsequences are retained from the decomposition.

Start with `rank=5`, following the SST rule of thumb, and compare a few smaller or larger values when validation data is available. A useful rank is large enough to retain recurring structure but not so large that noise-dominated directions control the comparison. Multivariate data does not automatically require `rank` to equal the number of channels; rank describes the joint block-Hankel structure.

For `rsvd`, `random_rank` controls the internal approximation dimension. If unset, the implementation uses:

```python
random_rank = min(rank + 10, window_length, n_windows)
```

Here `random_rank` is the sampled dimension, so the oversampling amount is `random_rank - rank`. The default therefore adds up to ten directions, bounded by the matrix dimensions. This agrees with the randomized-SVD literature's common oversampling range of roughly 5 to 10. Increasing it can improve reliability when singular values decay slowly, at the cost of more matrix products and orthogonalization.

## Runtime and Memory

Let:

- `T` be the number of samples.
- `d` be the number of channels (`d=1` for `SST` and `ESST`).
- `w = window_length`.
- `n = n_windows`.
- `k = rank`.
- `q = random_rank`.
- `s = scoring_step`.

Each subspace algorithm exposes two helper methods from `SingularSubspaceAlgorithm`:

```python
total_region, matrix_region = detector.covered_regions()
runtime_seconds, runtime_std = detector.estimate_runtime(signal, steps=30)
```

`covered_regions()` returns:

- `matrix_region = window_length + n_windows - 1`: the number of samples covered by one Hankel matrix.
- `total_region = matrix_region + lag`: the minimum region needed to compare the past and future Hankel matrices.

The signal must be longer than `total_region`. Scores before `detector.first_score_position` are not valid change scores and remain zero.

The number of evaluated positions is approximately:

```text
(T - total_region) / scoring_step
```

Increasing `scoring_step` is the simplest runtime lever: `scoring_step=2` evaluates roughly half as many positions, and `scoring_step=10` evaluates roughly one tenth as many positions.

The trajectory matrix has shape approximately:

```text
univariate:  w x n
multivariate: (d * w) x n
```

So memory and decomposition cost grow with `window_length`, `n_windows`, and the number of channels. Multivariate methods can be much more expensive because the row dimension grows with `d * w`.

`estimate_runtime()` warms up the algorithm on the first processable slice, measures `steps` repeated transformations of that slice, and scales the mean and standard deviation to the number of evaluated positions in the full signal. This scaling accounts for `scoring_step`. The warm-up is important because several kernels are JIT compiled on first use. The estimate is machine- and configuration-specific, so it is most useful for comparing candidate settings on the machine that will run them.

Example:

```python
from changepoynt.algorithms.sst import SST

slow = SST(window_length=300, method="rsvd", use_fast_hankel=False)
fast = SST(window_length=300, method="rsvd", use_fast_hankel=True)

slow_seconds, slow_std = slow.estimate_runtime(signal, steps=30)
fast_seconds, fast_std = fast.estimate_runtime(signal, steps=30)

print(f"slow: {slow_seconds:.2f}s +/- {slow_std:.2f}s")
print(f"fast: {fast_seconds:.2f}s +/- {fast_std:.2f}s")
```

Use these estimates comparatively: test several combinations of `window_length`, `n_windows`, `method`, and `use_fast_hankel`, then choose the cheapest setup that still gives usable scores. Increase `steps` for a more stable estimate, especially if individual transformations are very fast.

For a final wall-clock check:

```python
from time import perf_counter

start = perf_counter()
score = detector.transform(signal)
elapsed = perf_counter() - start
```

## When to Use Fast Hankel

Set `use_fast_hankel=True` to represent the Hankel matrix implicitly and perform its matrix products with FFT-based convolution. This is valuable for iterative methods such as IKA and randomized SVD because they need matrix products but do not necessarily need every matrix entry materialized.

Practical rule:

- Keep `use_fast_hankel=False` as the simple baseline for small matrices.
- Benchmark `True` and `False` once matrix products dominate the runtime.
- Expect the crossover to depend on `window_length`, `n_windows`, channel count, method, and hardware.

The Fast SST experiments found a crossover around a window length of 200 for their evaluated IKA-SST configurations and around 300 for randomized SVD. These are useful test points, not universal thresholds.

Fast Hankel is not available for every method in the current implementation:

| Algorithm | Fast Hankel support |
| --- | --- |
| `SST` | Supported for `rsvd`, `ika`, `weighted`, and `symmetric`. |
| `ESST` | Supported for `rsvd`; not supported for `fbrsvd`. |
| `MSST` | Supported for `rsvd`, `ika`, `weighted`, and `symmetric`. |
| `MESST` | Supported for `rsvd`. |

For univariate `SST` and `ESST`, `use_fast_hankel=True` cannot be combined with `mitigate_offset=True`.

!!! note "Current MSST behavior"
    The current `MSST.transform()` implementation always selects its fast block-Hankel representation, even when `use_fast_hankel=False`. Until that implementation detail is changed, the flag cannot be used to compare materialized and fast Hankel modes for `MSST`.

## Decomposition Methods

The `method` parameter controls both how the Hankel matrix is decomposed and, for some SST variants, how the score is formed. Available methods differ by algorithm.

Let `m = window_length` for univariate data, `m = n_channels * window_length` for multivariate data, and `n = n_windows`.

| Method | Algorithms | What it does | Computational character |
| --- | --- | --- | --- |
| `svd` | `SST` | Uses the Rayleigh-Ritz SST implementation for the past subspace and a power method for the future direction. | Deterministic subspace approximation; requires materialized matrices. |
| `naive` | `SST` | Computes full SVDs for both matrices and compares their subspaces. | Reference method with dense-SVD cost on the order of `m * n * min(m, n)` per matrix. |
| `naive updated` | `SST` | Computes full SVDs but uses the leading future direction in the SST-style score. | Similar dense cost to `naive`; useful as a reference. |
| `ika` | `SST`, `MSST` | Uses power iteration and a Lanczos/Krylov approximation with feedback from the preceding score position. | Avoids a full SVD. Cost grows with the number of matrix products and `lanczos_rank`; approximate and sequentially stateful. |
| `rsvd` | All four | Samples a low-dimensional range and performs a small SVD. The package uses two subspace iterations. | Dense work scales roughly with matrix products of size `m x n` by `random_rank`; benefits directly from fast structured products. |
| `fbrsvd` | `SST`, `ESST` | Uses the `fbpca` randomized implementation on a materialized matrix. | Approximate and often cheaper than a full SVD, but not compatible with fast Hankel here. |
| `weighted` | `SST`, `MSST` | Uses several future singular vectors and weights their scores by singular values. | More decomposition work than the leading-direction score; can represent richer future structure. |
| `symmetric` | `SST`, `MSST` | Compares both subspaces symmetrically. | Performs low-rank work for both directions and can reduce score directionality. |

Use this decision tree:

1. For `ESST` or `MESST`, use `method="rsvd"`; it is the only method shared by both implementations.
2. For `SST` or `MSST`, compare `ika` and `rsvd` when runtime matters.
3. Use `rsvd` when you want explicit control over the sampled rank and compatibility with fast Hankel products.
4. Use the dense or naive SST methods on small problems as references for approximation behavior.
5. Try `weighted` or `symmetric` only when their different score definitions match the behavior you want to emphasize.

Randomized methods draw a new random matrix during decomposition. Small run-to-run differences are therefore expected. For reproducible experiments, set NumPy's random seed before `transform()` and record the package version and all detector parameters.

### IKA Parameters

`lanczos_rank` controls the size of IKA's Krylov approximation. If it is omitted, the package derives an even value close to `2 * rank`. Raising it can improve the approximation but requires more Krylov work. `feedback_noise_level` adds noise to the feedback direction; it can prevent a fully recycled direction from becoming too rigid, but it also makes results stochastic.

## Interpreting the Result

`transform()` returns a score, not final change-point indices. A threshold and any peak-spacing rule remain application decisions.

- Ignore the zero-padded prefix before `first_score_position` when estimating a threshold.
- Estimate normal score variation from representative change-free data when possible.
- Treat nearby elevated samples as one event if the application expects one transition rather than many adjacent change points.
- Compare detection timing in physical units after accounting for the sample rate and the context required by the detector.
- Validate threshold and parameters together; changing the window or decomposition can change the score distribution.

These sliding comparisons require samples on both sides of the aligned change position. They can be updated as data arrives, but a score cannot be available before the detector has accumulated the required future context. The ESST paper explicitly notes window length as a contributor to minimum detection delay.

## Examples

### SST with IKA

```python
from changepoynt.algorithms.sst import SST

detector = SST(
    window_length=120,
    method="ika",
    rank=5,
    scoring_step=2,
)
score = detector.transform(signal)
```

### ESST with Randomized SVD

```python
from changepoynt.algorithms.esst import ESST

detector = ESST(
    window_length=120,
    method="rsvd",
    rank=5,
    random_rank=15,
)
score = detector.transform(signal)
```

### MESST for Multivariate Signals

```python
from changepoynt.algorithms.messt import MESST

detector = MESST(
    window_length=120,
    method="rsvd",
    rank=5,
    scoring_step=2,
)
score = detector.transform(multichannel_signal)
```

### Comparing Window Scales

```python
from changepoynt.algorithms.esst import ESST

results = {}
for window_length in (60, 120, 240):
    detector = ESST(window_length=window_length, method="rsvd")
    runtime, runtime_std = detector.estimate_runtime(signal)
    results[window_length] = {
        "score": detector.transform(signal),
        "first_score_position": detector.first_score_position,
        "estimated_runtime": runtime,
        "estimated_runtime_std": runtime_std,
    }
```

Compare only valid portions of the scores because each window length has a different `first_score_position`.

### Trying Fast Hankel

```python
from changepoynt.algorithms.sst import SST

detector = SST(
    window_length=300,
    method="ika",
    use_fast_hankel=True,
    scoring_step=5,
)
score = detector.transform(signal)
```

## Common Problems

| Symptom | What to check |
| --- | --- |
| The score is noisy everywhere | Increase `window_length` or `n_windows`, reduce `rank`, and check whether scaling amplifies a nearly constant signal. |
| Short changes disappear | Reduce `window_length` and verify that `scoring_step` is not skipping the relevant scale. |
| A multivariate score follows one channel | Inspect channel scaling and decide whether equalized ranges or original amplitudes are appropriate. |
| The signal is too short | Reduce the context parameters or use `covered_regions()` to determine the minimum required length. |
| Approximate results vary between runs | Fix NumPy's random seed and compare a larger `random_rank`. |
| `use_fast_hankel=True` raises an error | Check the method compatibility table and disable `mitigate_offset` for univariate SST/ESST. |

## Sources and Further Reading

- Tsuyoshi Ide and Keisuke Inoue, [Knowledge discovery from heterogeneous dynamic systems using change-point correlations](https://epubs.siam.org/doi/10.1137/1.9781611972757.63), SIAM International Conference on Data Mining, 2005.
- Tsuyoshi Ide and Koji Tsuda, [Change-point detection using Krylov subspace learning](https://epubs.siam.org/doi/10.1137/1.9781611972771.54), SIAM International Conference on Data Mining, 2007.
- Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp, [Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions](https://arxiv.org/abs/0909.4061), SIAM Review, 2011.
- Sarah Boelter, Lucas Weber, Richard Lenz, Brian Glass, and Maria Gini, [Fault Prediction in Planetary Drilling Using Subspace Analysis Techniques](https://ntrs.nasa.gov/citations/20250002705), IAS-19, 2025.
- Lucas Weber and Richard Lenz, [Accelerating Singular Spectrum Transformation for Scalable Change Point Detection](https://doi.org/10.1109/ACCESS.2025.3640386), IEEE Access, 2025.
