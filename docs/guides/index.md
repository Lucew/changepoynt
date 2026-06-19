# User Guides

The algorithm overview lists what is available. The guides focus on choosing and configuring an algorithm for a particular kind of change.

## Subspace Methods

[Tuning subspace methods](tuning-subspace-methods.md)

Use this guide for a coarse-to-fine parameter search, runtime estimation, decomposition complexity, and fast Hankel decisions for `SST`, `ESST`, `MSST`, and `MESST`.

[Fast SST and IKA-SST](../fast-sst.md)

Use this focused guide when large Hankel matrices dominate the runtime of SST-style methods.

## Visualization

[Visualizing change scores](visualizing-scores.md)

Plot a signal and its change score together, using background shading to align high-score regions with the original data.

## Experimental Streaming

[Buffered stream processing](experimental-buffered-streaming.md)

Experiment with repeatedly applying the existing batch `transform()` interface to a fixed-size rolling buffer and emitting the newly available delayed score.

## Adding Further Guides

Future guides should be grouped by method family rather than by individual class. Suitable groups for the current package are:

- Density-ratio methods: `ULSIF` and `RuLSIF`.
- Matrix-profile segmentation: `FLUSS` and `FLOSS`.
- Probabilistic methods: `BOCPD`.

Each guide can compare related algorithms, explain their shared parameters, and link to the individual classes in the API reference.
