# Frequently Asked Questions (FAQ)

## How can I set the window size?
Unfortunately, there is no one answer to this question. Setting the window size essentially specifies the region of
interest along the time axis, where the algorithm looks for changes. The naive solution is to compute multiple window
sizes and scan through the all maxima. Alternatively, try to get access to domain knowledge. If you know in which time
frame your changes are normally visible in the time series, we recommend setting the window size a little larger than
the duration of your change (1.5x or 2x). If you have periodic signals, you could use the FFT or cross-correlation
function and set the window size to a multiple of it for checking frequency changes or using the periodicity directly
for finding abnormal periods.

## How can I get discrete change points from the score?
While this is also a challenging question, there is literature for finding anomalies from anomaly scoring. These
approaches also apply to our problem. Additionally, you might want to take a look at what packages like 
[PyOD](https://github.com/yzhao062/pyod) and [PyThresh](https://github.com/KulikDM/pythresh) recommend.

## The algorithms are slow for large window sizes, how can I speed them up?
> TL;DR: Consider using `use_fast_hankel = True` for large window sizes.
> Consider increasing `scoring_step` to make larger steps through the signal (`scoring_step=2` is twice as fast).
> 
For the algorithms that build up on the decomposition of time series Hankel matrices (like 
[SST](https://github.com/Lucew/changepoynt/blob/master/changepoynt/algorithms/sst.py) and 
[ESST](https://github.com/Lucew/changepoynt/blob/master/changepoynt/algorithms/esst.py)), we created an
signficantly more efficient algorithm (see [the code for the paper](https://github.com/Lucew/approximate_hankel) for
reference).

This efficient algorithm can be used when specifying the option `use_fast_hankel = True` in the 
[SST](https://github.com/Lucew/changepoynt/blob/master/changepoynt/algorithms/sst.py) and
[ESST](https://github.com/Lucew/changepoynt/blob/master/changepoynt/algorithms/esst.py). Empirically, we saw
improvements for window size larger than 200 for the SST and larger than 400 for the ESST. These boundaries depend on 
your system parameters, like available libraries and cores. For Window sizes smaller than this threshold, the naive
implementation is typically faster. For tiny window size (~20) the naive implementation will outperform the
efficient algorithm by around a magnitude.

Additionally, consider increasing `scoring_step` to make larger steps through the signal.
The SST computes a score at every sample by definition. For large window sizes the processes windows will largely overlap,
producing very correlated, unnecessary measurements. By increasing scoring step you can decrease the overlap,
significantly reducing the runtime. For example, `scoring_step=2` skips every other sample and the computation runs 
twice as fast.

Other than that, you can always use the multiprocessing module in python, which works well for the methods in this
package. Please refrain from using the threading module due to the Global interpreter lock and our JIT compiler.