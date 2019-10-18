# ClimateChangeFromMachineLearningResearch

## Usage


```python
from experiment_impact_tracker.compute_tracker import ImpactTracker, get_flop_count_tensorflow
tracker = ImpactTracker("{}_{}".format(args.model_name,args.seed))
tracker.launch_impact_monitor()
```

