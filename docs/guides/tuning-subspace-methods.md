# Tuning Subspace Methods

This guide focuses on tuning `SST`, `ESST`, `MSST`, and `MESST`. A practical order is:

1. `window_length`
2. `n_windows`
3. `lag`
4. `rank`

This order reflects how directly each parameter controls the temporal scale of the comparison. It is a useful workflow, not a universal ranking: unusual signals may still require revisiting an earlier choice.

Keep the decomposition method and runtime settings fixed while tuning signal behavior. Once the score is useful, optimize its execution without changing several numerical approximations at once.

## Before Tuning

Choose representative data containing:

- More than one change, when available.
- Normal sections with realistic noise and operating variation.
- The sampling rate and approximate duration of relevant patterns.

For notation, let:

- `T` be the number of samples.
- `d` be the number of channels; `d = 1` for `SST` and `ESST`.
- `w = window_length`.
- `n = n_windows`.
- `k = rank`.
- `l = random_rank` for randomized SVD.
- `s = scoring_step`.

One trajectory matrix has approximately `m = d * w` rows and `n` columns. It covers:

```text
matrix_region = w + n - 1
total_region  = w + n - 1 + lag
```

Inspect these values directly:

```python
total_region, matrix_region = detector.covered_regions()
print(total_region, matrix_region, detector.first_score_position)
```

## Fast Initial Runtime Pass

For an initial feasibility test, vary only `window_length`. Leave `n_windows`, `lag`, and `rank` at their defaults so they remain coupled to the window as intended by each algorithm. Use a fast-Hankel-compatible decomposition and a `scoring_step` below the window length.

!!! tip "Estimate runtime before processing the full signal"
    `SST`, `ESST`, `MSST`, and `MESST` inherit `estimate_runtime()` from the `SingularSubspaceAlgorithm` base class. Call it with the real signal and the candidate detector configuration:

    ```python
    estimated_seconds, estimated_std = detector.estimate_runtime(
        signal,
        steps=30,
    )
    ```

    The utility warms up first-use JIT compilation, times a processable slice repeatedly, and scales the measurement to the signal length and `scoring_step`. It returns the estimated runtime and its estimated standard deviation in seconds. Use it to reject expensive configurations before running `transform()` on the complete signal.

A useful coarse starting point is:

```text
scoring_step = max(1, window_length // 5)
```

This is deliberately coarse. If short events matter, use a smaller fraction such as `window_length // 10` or derive the step from the shortest event that must remain visible.

```python
from changepoynt.algorithms.esst import ESST

runtime_estimates = {}

for window_length in (60, 120, 240):
    scoring_step = max(1, window_length // 5)
    detector = ESST(
        window_length=window_length,
        method="rsvd",
        use_fast_hankel=True,
        scoring_step=scoring_step,
    )
    runtime_estimates[window_length] = detector.estimate_runtime(
        signal,
        steps=30,
    )
```

This quickly shows whether the desired temporal scales are computationally realistic. It also captures the coupled growth of the defaults: increasing `window_length` increases the matrix rows and also increases default `n_windows` and `lag`.

Because `scoring_step` changes with the window in this pass, these estimates describe practical coarse configurations rather than the isolated asymptotic cost of `window_length`. Use a common step later when making a controlled runtime comparison.

### What `scoring_step` Outputs

With `scoring_step=1`, the algorithm evaluates every possible position. With a larger step, it evaluates only every `scoring_step` samples.

The missing positions are not interpolated. The implementation fills the interval between evaluation positions by holding the computed score constant. In other words, the output is a centered zero-order hold: larger steps produce wider piecewise-constant score plateaus.

```text
evaluated:  a           b           c
output:     a a a a a   b b b b b   c c c c c
```

The number of evaluations is approximately:

```text
(T - total_region) / scoring_step
```

Fast Hankel reduces the cost of the matrix products, while `scoring_step` reduces how often those products and decompositions are performed. Together they make large-window screening much cheaper.

Use these coarse settings only to compare window scales and runtime. Once a promising window is found, reduce `scoring_step` and confirm that peak shape, timing, and short events remain acceptable.

## 1. Tune `window_length`

`window_length` is the main scale parameter. Each trajectory-matrix column contains this many consecutive samples, so the window must be long enough to represent the shape that should remain stable within one regime.

### Effect on the Score

A larger window gives each subspace more temporal context. It tends to smooth the score and represent longer periodic or transient patterns, but it can broaden peaks and dilute changes shorter than the window.

A smaller window is more local and can preserve short changes, but it may represent only part of a recurring pattern. Scores can then react to noise or produce several peaks around one transition.

Because default `n_windows` and `lag` are derived from `window_length`, changing the window also changes the amount and separation of data used by the complete comparison.

### Effect on Runtime

Increasing the window adds trajectory-matrix rows. With algorithm defaults, it also increases `n_windows`, so both matrix dimensions grow together. Dense decomposition cost therefore grows much faster than linearly with `window_length`.

Fast Hankel changes the matrix-product scaling, but larger windows still require longer FFTs, larger low-rank projections, and more context. This is why `window_length` is the most important parameter for both scoring behavior and initial runtime tests.

### How to Tune It

Convert physical duration into samples:

```python
window_length = round(duration_seconds * sampling_rate_hz)
```

Good initial candidates are values around the expected duration, for example `0.5x`, `1x`, and `2x`. For periodic data, begin around one period and compare nearby multiples.

**Signs that the window is too small**

- Recurring structure is only partially represented.
- Scores tend to react to short fluctuations and noise.
- Several peaks may appear around one broader transition.

**Signs that the window is too large**

- Short changes can be diluted.
- Peaks become wider and require more surrounding data.
- Short score features are lost even after reducing `scoring_step`.

Tune on a coarse grid first. Do not search every integer window length: neighboring values usually produce highly related representations.

```python
from changepoynt.algorithms.esst import ESST

scores = {}
for window_length in (60, 120, 240):
    scoring_step = max(1, window_length // 5)
    detector = ESST(
        window_length=window_length,
        method="rsvd",
        use_fast_hankel=True,
        scoring_step=scoring_step,
    )
    scores[window_length] = detector.transform(signal)
```

Use the same `method` and `rank` for every candidate. Here the step scales with the window to keep initial runs inexpensive; repeat the best candidates with a smaller, common step before comparing detailed peak shapes. Compare only valid score regions because `first_score_position` changes with the window.

## 2. Tune `n_windows`

`n_windows` is the number of neighboring subsequences used to estimate local structure. It controls how many examples contribute to each trajectory matrix and, together with `window_length`, how much time one matrix covers.

### Effect on the Score

More windows provide more examples of local behavior. This can stabilize the estimated subspace and suppress score variation caused by noise, but a long matrix can mix structures that should remain local and can smear rapid changes.

Fewer windows make the representation more local and responsive. If too few columns are available, however, the low-rank structure is poorly constrained and randomized decompositions have less room for rank plus oversampling.

### Effect on Runtime

Increasing `n_windows` adds matrix columns. Materialized storage grows as `O(d * window_length * n_windows)`, and every dense or randomized matrix product becomes more expensive. It also increases `matrix_region` and therefore the minimum amount of signal needed.

If `lag` is left as `None`, changing `n_windows` also changes the derived lag. During a focused `n_windows` sweep, explicitly decide whether to preserve that coupling or hold `lag` fixed.

### How to Tune It

Package defaults are:

- `SST` and `MSST`: `n_windows = window_length`.
- `ESST` and `MESST`: `n_windows = window_length // 2`.

Keep these defaults for the first window-length sweep. Then change `n_windows` only when the score suggests a reason.

**Use fewer windows when**

- The normal structure changes quickly.
- A long matrix mixes separate local behaviors.
- The score is broader than the behavior being detected.

The ESST paper specifically recommends `n_windows < window_length` when time-series structure changes rapidly.

**Use more windows when**

- Normal behavior is variable and needs more examples.
- The estimated subspace changes too much from one score position to the next.
- The score is noisy despite a reasonable `window_length`.

A useful local sweep is often the default value plus one smaller and one larger candidate rather than an independent wide search. Keep `window_length`, `lag`, `rank`, `method`, and `scoring_step` fixed when comparing the detailed score effect.

## 3. Tune `lag`

`lag` controls the displacement between the past and future trajectory matrices. It determines how separated the compared behaviors are.

### Effect on the Score

A smaller lag compares nearby, often overlapping behavior. This can localize abrupt transitions but may produce weak contrast when both matrices contain many of the same samples.

A larger lag reduces overlap and can reveal gradual changes more clearly. If it is too large, the score can respond to unrelated drift or compare operating states that no longer belong to one local transition.

Lag changes the context needed to produce a score, but the implementation offsets the output to align it with the comparison region. Evaluate both peak strength and timing rather than assuming that a larger lag simply shifts the returned score by the same amount.

### Effect on Runtime

Lag does not change the trajectory-matrix dimensions, so the cost of one decomposition is essentially unchanged. A larger lag increases `total_region`; for a fixed signal, that leaves slightly fewer valid evaluation positions and may slightly reduce total runtime.

Its main computational cost is therefore data availability and latency, not matrix algebra. This differs from `window_length` and `n_windows`, which directly enlarge each decomposition.

### How to Tune It

Package defaults are:

- `SST` and `MSST`: `lag = max(n_windows // 3, 1)`.
- `ESST` and `MESST`: `lag = n_windows`.

**Use a smaller lag when**

- Changes must be detected with less accumulated future context.
- The signal evolves quickly and distant comparisons mix unrelated behavior.
- A transition is sharp and local comparisons already separate both regimes.

**Use a larger lag when**

- Past and future matrices overlap too strongly.
- The change unfolds gradually.
- Nearby comparisons are dominated by normal local variation.

Tune lag after `window_length` and `n_windows`, because its meaning depends on the temporal extent of those matrices. Keep all matrix-shape and decomposition settings fixed during the comparison.

## 4. Tune `rank`

`rank` is the number of dominant structural directions retained for scoring. The SST literature uses `rank=5` as a low-rank rule of thumb, which is also the package default.

### Effect on the Score

A smaller rank emphasizes only the dominant recurring structures. This can suppress noise, but it can miss changes in weaker oscillations or joint channel relationships.

A larger rank includes more structure and can detect subtler changes. Once noise-dominated directions are included, however, the score can become less stable and less selective.

Rank describes the joint block-Hankel structure, not the number of channels. A multivariate signal with many correlated channels can still be low rank.

### Effect on Runtime

For `rsvd`, increasing `rank` normally increases `random_rank`, so projection, orthogonalization, and the small SVD all become more expensive. For `ika`, the default `lanczos_rank` grows to roughly twice `rank`, increasing Krylov work.

The dense `naive` methods compute full SVDs, so lowering the retained score rank does not reduce their main decomposition cost nearly as much. Rank is a stronger runtime control for truncated and iterative methods.

### How to Tune It

Start with `5` and compare a small set such as `2`, `3`, `5`, and `8` when needed.

**Use a smaller rank when**

- The signal consists of one or two simple recurring components.
- Higher-rank scores are unstable or react strongly to noise.
- The leading components already capture the repeatable behavior.

**Use a larger rank when**

- Several oscillations, trends, or joint channel patterns matter.
- A lower rank misses repeatable structure.
- Important changes appear only outside the leading directions.

Keep `rank` well below `min(d * window_length, n_windows)`. Randomized methods also need room for oversampling. By default:

```python
random_rank = min(rank + 10, window_length, n_windows)
```

The randomized-SVD literature commonly recommends 5 to 10 oversampled directions. If results vary strongly between runs or singular values decay slowly, increase `random_rank` before increasing the score rank itself.

## Runtime Estimation

Total runtime is roughly the number of evaluated positions multiplied by the decomposition cost at one position. For materialized matrices, storage is proportional to `m * n`; multivariate methods increase `m` to `d * w`.

Use the package estimator for actual comparisons on the target machine:

```python
seconds, standard_deviation = detector.estimate_runtime(
    signal,
    steps=30,
)
```

The estimator performs a warm-up so first-use JIT compilation is not included. Treat it as a configuration estimate, then time one full final run.

## Tune `scoring_step`

`scoring_step` is usually the simplest runtime control after selecting the matrix dimensions.

### Effect on the Score

The algorithm evaluates one position per step and holds that value across the surrounding output interval. Larger steps therefore produce wider constant plateaus; they do not reconstruct the skipped positions by interpolation.

If the step approaches or exceeds the width of an important score feature, that feature can be skipped, flattened, or shifted to the nearest evaluated position. Keep the final step comfortably below both `window_length` and the shortest event that must remain distinguishable.

### Effect on Runtime

A value of `5` evaluates about one fifth as many positions as `1`, so runtime is approximately inversely proportional to the step. Matrix size and per-evaluation decomposition cost do not change.

### How to Tune It

Use a coarse step while exploring expensive configurations, then lower it for the final detector:

```python
from changepoynt.algorithms.esst import ESST

coarse = ESST(window_length=240, method="rsvd", scoring_step=10)
final = ESST(window_length=240, method="rsvd", scoring_step=2)
```

For initial window tests, derive a coarse step from the window. For final validation, lower the step until further reductions no longer change the event-level conclusions.

## Choose a Decomposition

Let the trajectory matrix have shape `m x n` and let `l` be the randomized approximation size.

### Effect on the Score

`svd`, `ika`, `rsvd`, and `fbrsvd` aim at related low-rank SST quantities but use different deterministic or approximate routes. IKA and randomized methods can produce small numerical or run-to-run differences. `weighted` and `symmetric` also change the score definition, so they should not be selected on runtime alone.

### Effect on Runtime

The decomposition determines how many dense or structured matrix products are required and whether the full matrix must be materialized. Its interaction with `rank`, `random_rank`, `lanczos_rank`, and fast Hankel usually dominates the cost of one evaluated position.

`naive` and `naive updated` (`SST` only)
: Compute dense SVDs of both matrices. Dense SVD costs approximately `O(m * n * min(m, n))` per matrix and stores `O(m * n)` values. Use these as small-problem references, not as the first choice for large scans.

`svd` (`SST` only)
: Uses the package's Rayleigh-Ritz subspace approximation and power iteration. It requires materialized Hankel matrices and is mainly useful as a deterministic SST-style reference.

`rsvd` (all four algorithms)
: Samples a subspace of size `l`, performs two power/subspace iterations in this package, and decomposes a smaller matrix. Dense work is dominated by products on the order of `O(m * n * l)` per pass, plus lower-dimensional orthogonalization and SVD work. It is approximate, scales with `random_rank`, and supports fast Hankel products.

`fbrsvd` (`SST` and `ESST`)
: Uses the `fbpca` randomized implementation on materialized matrices. It can be effective for moderate dense matrices but does not support `use_fast_hankel=True` here.

`ika` (`SST` and `MSST`)
: Uses power iteration and a Lanczos/Krylov approximation with feedback from the preceding position. Cost depends on the number of structured matrix products and `lanczos_rank`, whose default is close to `2 * rank`. IKA avoids a full dense SVD and is attractive when sequential scoring and fast Hankel products are suitable.

`weighted` and `symmetric` (`SST` and `MSST`)
: These are different randomized-SVD score definitions, not merely faster decompositions. They use more future directions or both comparison directions, so choose them for their score behavior and expect additional low-rank work.

For runtime-sensitive `SST` or `MSST`, compare `ika` and `rsvd`. For `ESST`, use `rsvd` when enabling fast Hankel; `MESST` currently supports only `rsvd`.

## Enable Fast Hankel Products

### Effect on the Score

Fast Hankel changes how Hankel products are computed, not the intended scoring formula. Expect small floating-point or approximation differences, especially with randomized and iterative decompositions, but not a deliberately different score definition. Validate peak ordering and timing against the materialized version on a representative subset.

### Effect on Runtime

With `use_fast_hankel=False`, the package materializes trajectory matrices. A dense Hankel matrix-vector product costs `O(m * n)` and matrix storage costs `O(m * n)`.

Fast Hankel represents the structure implicitly and uses FFT-based convolution. For a univariate Hankel matrix, one structured product is approximately `O((w + n) log(w + n))` instead of `O(w * n)`, with linear rather than matrix-sized structural storage. Block-Hankel methods repeat related work across channels.

FFT setup and bookkeeping have overhead, so fast Hankel is not automatically faster for small matrices. The Fast SST experiments observed crossover points around window length 200 for their IKA configurations and around 300 for randomized SVD. Use these as places to begin benchmarking, not fixed thresholds.

```python
from changepoynt.algorithms.sst import SST

regular = SST(window_length=300, method="rsvd")
fast = SST(window_length=300, method="rsvd", use_fast_hankel=True)

regular_time, _ = regular.estimate_runtime(signal)
fast_time, _ = fast.estimate_runtime(signal)
```

Compatibility:

- `SST`: `ika`, `rsvd`, `weighted`, and `symmetric`.
- `ESST`: `rsvd`.
- `MSST`: all currently exposed methods; its current transform path always requests the fast block representation.
- `MESST`: `rsvd`.
- Univariate `mitigate_offset=True` cannot be combined with fast Hankel.

## A Practical Search

1. Select one algorithm and a fast-Hankel-compatible decomposition.
2. Enable fast Hankel and set `scoring_step` from each candidate window, initially around `window_length // 5`.
3. Estimate runtime while sweeping `window_length` on a coarse physical-scale grid.
4. Re-run promising windows with a smaller common `scoring_step` and compare score behavior.
5. Tune `n_windows` around its algorithm default while holding the other parameters fixed.
6. Tune `lag` around its default after the matrix span is fixed.
7. Compare a small set of ranks.
8. Compare decomposition methods and regular versus fast Hankel where supported.
9. Reduce `scoring_step` until event-level conclusions stabilize, then validate full runtime and scoring on held-out data.

Record the full detector configuration, package version, sample rate, and NumPy random seed. Randomized decompositions and IKA feedback can otherwise produce small run-to-run differences.

## References

- Tsuyoshi Ide and Keisuke Inoue, [Knowledge discovery from heterogeneous dynamic systems using change-point correlations](https://epubs.siam.org/doi/10.1137/1.9781611972757.63), 2005.
- Tsuyoshi Ide and Koji Tsuda, [Change-point detection using Krylov subspace learning](https://epubs.siam.org/doi/10.1137/1.9781611972771.54), 2007.
- Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp, [Finding structure with randomness](https://arxiv.org/abs/0909.4061), 2011.
- Sarah Boelter et al., [Fault Prediction in Planetary Drilling Using Subspace Analysis Techniques](https://ntrs.nasa.gov/citations/20250002705), 2025.
- Lucas Weber and Richard Lenz, [Accelerating Singular Spectrum Transformation for Scalable Change Point Detection](https://doi.org/10.1109/ACCESS.2025.3640386), 2025.
