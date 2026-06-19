# Algorithms

`changepoynt` implements several change point detection approaches from subspace estimation, density-ratio estimation, and time-series segmentation.

| Algorithm | Source | Status |
| --- | --- | --- |
| [SST](algorithms/sst.md) | [Ide, 2005](https://epubs.siam.org/doi/abs/10.1137/1.9781611972757.63) | Stable |
| [Fast SST](fast-sst.md) | [Weber et al., 2025](https://doi.org/10.1109/ACCESS.2025.3640386) | Stable |
| [IKA-SST](algorithms/sst.md) | [Ide, 2007](https://epubs.siam.org/doi/abs/10.1137/1.9781611972771.54) | Stable |
| [Fast IKA-SST](fast-sst.md) | [Weber et al., 2025](https://doi.org/10.1109/ACCESS.2025.3640386) | Stable |
| [Multivariate (IKA-)SST (MSST)](algorithms/msst.md) |  | Stable |
| [ESST](algorithms/esst.md) | [Boelter & Weber et al., 2025](https://ntrs.nasa.gov/citations/20250002705) | Stable |
| [Multivariate ESST (MESST)](algorithms/messt.md) |  | Stable |
| [RuLSIF](algorithms/rulsif.md) | [Liu et al.](https://www.sciencedirect.com/science/article/pii/S0893608013000270) | Stable |
| [uLSIF](algorithms/ulsif.md) | [Liu et al.](https://www.sciencedirect.com/science/article/pii/S0893608013000270) | Stable |
| KLIEP | [Liu et al.](https://www.sciencedirect.com/science/article/pii/S0893608013000270) | Planned |
| [ClaSP](algorithms/clasp.md) | [Ermshaus et al.](https://link.springer.com/article/10.1007/s10618-023-00923-x) | Deactivated |
| [FLUSS](algorithms/fluss.md) | [Gharghabi et al.](https://ieeexplore.ieee.org/abstract/document/8215484) | Stable |
| [FLOSS](algorithms/floss.md) | [Gharghabi et al.](https://ieeexplore.ieee.org/abstract/document/8215484) | Unavailable |
| [BOCPD](algorithms/bocpd.md) | [Adams et al.](https://arxiv.org/abs/0710.3742) | Experimental |
| [MovingWindow](algorithms/moving-window.md) | Wu & Keogh | Stable baseline |
| [ZERO](algorithms/zero.md) | van den Burg & Williams | Stable baseline |
| [Subspace Identification](algorithms/subspace-identification.md) | Kawahara et al. | Unavailable |
| [TESST](algorithms/tesst.md) |  | Experimental |

## Choosing a Starting Point

For most one-dimensional signals, start with `ESST` or `SST`. If runtime becomes an issue for large window sizes, enable the accelerated Hankel implementation where supported.

For parameter choices, runtime trade-offs, and decomposition methods for `SST`, `ESST`, `MSST`, and `MESST`, see [Tuning Subspace Methods](guides/tuning-subspace-methods.md).

For streaming-style matrix-profile segmentation, consider `FLOSS`. For offline matrix-profile segmentation, consider `FLUSS`.

Density-ratio methods such as `ULSIF` and `RuLSIF` are useful when the change is best described as a distribution shift between neighboring windows.

## User Guides

The [user guides](guides/index.md) group practical advice by method family. This keeps the algorithm table useful as a catalog while allowing each guide to discuss related methods together.

| Method family | Guide |
| --- | --- |
| Subspace methods | [Tuning SST, ESST, MSST, and MESST](guides/tuning-subspace-methods.md) |
| Fast subspace methods | [Fast SST and IKA-SST](fast-sst.md) |
