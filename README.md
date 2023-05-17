# Python Changepoint Detection (changepoynt)

 

This is the repository hosting the **pip-installable** python package changepoynt. It implements several different change point detection techniques, while focusing mostly on "localized" algorithms, that could be run in an online fashion.

 

Current algorithms come from the field of:

* Statistics (Detection of Change in the statistical properties)

* Time Series Segmentation (Algorithms focused on comparing time series shap)

 

The package is aimed at execution performance (using JIT compilation and standing on the shoulders of giants like numpy and scipy) while also keeping code readable and maintainable. This includes comments as well as architectural choices. This might not be perfect, but we are trying!

 

All of our algorithms are implementations of a base changepoint detection interface and therefore are interchangeable. Currently, we are focused on shifting to the very common and existing sklearn interface of fit and predict. This enables our algorithms to be part of the standard sklearn pipeline for preprocessing.

# Examples

You can find example code within the examples folder of this repository. We also wanted to tease the
functionality using two different signals in order to show the capabilities of one of our recommended algorithms
[ESST](./changepoynt/algorithms/esst.py).

The first application is a simulated temperature of a component in a power plant during shutdown.
We artificially added a disturbance at the end of the shutdown, to show the capability of the algorithm to
detect a change even in case of another major change.

![image](./images/simulated_temperature_behavior.png)

The other application is for anomaly detection within periodic signals. The example is time series
34 from the
[Hexagon ML/UCR Time Series Anomaly Detection dataset](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/), where we
set the window size for the ESST to three times the estimated period in samples (estimated using maximum of FFT).

![image](./images/034_UCR_Anomaly_DISTORTEDInternalBleeding6_1500_3474_3629.png)
 
Both plots have been created using `changepoynt.algorithms.esst` and the plot function from 
`changepoynt.visualization.score_plotting`.

# Installation

You can install `changepoynt` from the common package index [PyPi](https://pypi.org/project/changepoynt/) using the following line with pip:

 

    pip install changepoynt

 

Please be aware, that we are currently in an alpha development phase, as this is part of a research project at the FAU Erlangen together with SIEMENS Energy developed by [me](https://www.cs6.tf.fau.eu/person/lucas-weber/). Nevertheless, we aim to be open-source and try our best to guarantee that all the code we use has very permissive licenses.

# Participating

We always love to get feedback or new ideas. If you have any of those, feel free to open an issue. We try to get back to you as soon as we can.

 

If you are an author of a paper in the field or have another algorithmic idea: Feel free to open a pull request. Currently, we are still working on the contribution guides. But if somebody already comes along and has an idea, we do not want to be in the way!