# ClimateChangeFromMachineLearningResearch

## Compatible Systems

Right now, we're only compatible with Linux systems running NVIDIA GPU's and Intel processors (which support RAPL). If you'd like support for your use-case or encounter missing/broken functionality on your system specs, please open an issue or better yet submit a pull request! It's almost impossible to cover every combination on our own!

## Installation

To install:

```bash
cd ClimateChangeFromMachineLearningResearch;
pip install -e .
```

## Usage

### Tracking
You just need to add a few lines of code!

```python
from experiment_impact_tracker.compute_tracker import ImpactTracker
tracker = ImpactTracker(<your log directory here>)
tracker.launch_impact_monitor()
```

This will launch a separate process (more like thread) that will gather compute/energy/carbon information in the background.

**NOTE: Because of the way python multiprocessing works, this process will not interrupt the main one UNLESS you periodically call the following. This will read the latest info from the log file and check for any errors that might've occured in the tracking process. If you have a better idea on how to handle exceptions in the tracking thread please open an issue or submit a pull request!!!** 

```python
info = tracker.get_latest_info_and_check_for_errors()
```

### Asserting certain hardware

It may be the case that you're trying to run two sets of experiments and compare emissions/energy/etc. In this case, you generally want to ensure that there's parity between the two sets of experiments. If you're running on a cluster you might not want to accidentally use a different GPU/CPU pair. To get around this we provided an assertion check that you can add to your code that will kill a job if it's running on a wrong hardware combo. For example:

```python
assert_gpus_by_attributes({ "name" : "GeForce GTX TITAN X"})
assert_cpus_by_attributes({ "brand": "Intel(R) Xeon(R) CPU E5-2640 v3 @ 2.60GHz" })
```

### Generating a LateX appendix from your data

You can generate an appendix that will aggregate certain info from the recorded data like this:

```bash
./scripts/create-compute-appendix experiment_results/mobilenet/raw/conv_mobilenet_v1_32_0 experiment_results/mobilenet/raw/conv_mobilenet_v1_32_1/ experiment_results/mobilenet/raw/conv_mobilenet_v1_32_2/ experiment_results/mobilenet/raw/conv_mobilenet_v1_32_3/ experiment_results/mobilenet/raw/conv_mobilenet_v1_32_4/ --experiment_set_names "Normal Convolutions (32 filters)"
```

This will generate a directory structure like the following, where each experiment in the set is given a number and the summary looks at the per experiment aggregates with standard error as well as cumulative usage.

```
<experiment_set_name>/0.pdf
<experiment_set_name>/1.pdf
<experiment_set_name>/2.pdf
<experiment_set_name>/3.pdf
<experiment_set_name>/4.pdf
<experiment_set_name>/summary.pdf
```

You can compare two sets of experiments as follows:

```bash
./scripts/create-compute-appendix experiment_results/mobilenet/raw/conv_mobilenet_v1_32_0 experiment_results/mobilenet/raw/conv_mobilenet_v1_32_1/ experiment_results/mobilenet/raw/conv_mobilenet_v1_32_2/ experiment_results/mobilenet/raw/conv_mobilenet_v1_32_3/ experiment_results/mobilenet/raw/conv_mobilenet_v1_32_4/ --experiment_set_names "Normal Convolutions (32 filters)" "Separable Convolutions (64 filters)" --compare_dirs experiment_results/mobilenet/raw/mobilenet_v1_64_0 experiment_results/mobilenet/raw/mobilenet_v1_64_1/ experiment_results/mobilenet/raw/mobilenet_v1_64_2/ experiment_results/mobilenet/raw/mobilenet_v1_64_3/ experiment_results/mobilenet/raw/mobilenet_v1_64_4/
```

This will create a file with the treatment effects of the first experiment versus the second experiment set (with standard error and p-value using Welch's t-test): 

```
<experiment_set_name_1>_v_<experiment_set_name_2>.pdf
```

