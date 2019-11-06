import pandas as pd
import os
import numpy as np

def load_rl_eval(impact_tracker_directory, aggregated_info):
    rl_eval = pd.read_csv(os.path.join(impact_tracker_directory, 'eval_test.csv'), skiprows=1)

    return {
        "AverageReturn" : np.mean(rl_eval["rmean"]),
        "AsymptoticReturn" : rl_eval["rmean"][len(rl_eval["rmean"])-1],
        "AverageReturnPerkWh" : np.mean(rl_eval["rmean"]) / aggregated_info["total_power"]
        }