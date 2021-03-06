__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import csv
import json
import os
import time
from glob import glob
import numpy as np
import pandas
from gym.core import Wrapper
import matplotlib.pyplot as plt
# Init seaborn
import scipy
import argparse
from experiment_impact_tracker.stats import run_test, get_average_treatment_effect
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--log-dirs', help='Log folder(s)', nargs='+', required=True, type=str)
parser.add_argument('--title', help='Plot title', default='Learning Curve', type=str)
parser.add_argument('--smooth', action='store_true', default=False,
                    help='Smooth Learning Curve')
args = parser.parse_args()

from baselines.common import plot_util as pu

results = pu.load_results(args.log_dirs)

grouping_fn = lambda r: "orig_arch" if "orig_arch" in r.dirname else "smaller_arch"
#TODO: fix this, seems like somwhere in ema_smoothing this starts to return nans and drops out :( 
# fig = pu.plot_results(results, average_group=True, resample=512, group_fn=lambda r: "orig_arch" if "orig_arch" in r.dirname else "smaller_arch", shaded_std=True)
# fig = pu.plot_results(results, average_group=True, group_fn=, shaded_std=True)
groups = {}

def default_xy_fn(r):
    x = np.cumsum(r.monitor.l)
    y = r.monitor.r
    return x,y

for result in results:
    if grouping_fn(result) not in groups:
        groups[grouping_fn(result)] = {}
    x, y = default_xy_fn(result)
    if  "average_returns" not in groups[grouping_fn(result)]:
        groups[grouping_fn(result)]["average_returns"] = []
    if "asymptotic_returns" not in groups[grouping_fn(result)]:
        groups[grouping_fn(result)]["asymptotic_returns"] = []
    groups[grouping_fn(result)]["average_returns"].append(np.mean(y))
    groups[grouping_fn(result)]["asymptotic_returns"].append(np.mean(y[-100:]))
# import pdb; pdb.set_trace()

for group in groups:
    print("Group {}".format(group))
    average_average_return = np.mean(groups[group]["average_returns"])
    stderr_average_return = scipy.stats.sem(groups[group]["average_returns"])# np.std(groups[group]["average_returns"])  / np.sqrt(len(groups[group]["average_returns"]))
    print("Average Returns {:.2f} \pm {:.2f}".format(average_average_return, stderr_average_return))    

    average_average_return = np.mean(groups[group]["asymptotic_returns"])
    stderr_average_return = scipy.stats.sem(groups[group]["asymptotic_returns"])# np.std(groups[group]["asymptotic_returns"])  / np.sqrt(len(groups[group]["asymptotic_returns"]))
    print("Asymptotic Returns {:.2f} \pm {:.2f}".format(average_average_return, stderr_average_return))    

    #TODO: significance testing, change to standard error

print("{} vs. {} Average returns".format(list(groups.keys())[0], list(groups.keys())[1]))
significant_difference, p = run_test('Welch t-test', np.array(groups[list(groups.keys())[0]]["average_returns"]), np.array(groups[list(groups.keys())[1]]["average_returns"]))
ate,  std_err = get_average_treatment_effect(np.array(groups[list(groups.keys())[0]]["average_returns"]), np.array(groups[list(groups.keys())[1]]["average_returns"]))
print("ATE : {:.2f} \pm {:.2f}".format(ate, std_err))
print("There was {}a significance difference with p={:.2f} and threshold .05.".format("" if significant_difference else "not ", p))
print("{} vs. {} asymptotic returns".format(list(groups.keys())[0], list(groups.keys())[1]))
significant_difference, p = run_test('Welch t-test', np.array(groups[list(groups.keys())[0]]["asymptotic_returns"]), np.array(groups[list(groups.keys())[1]]["asymptotic_returns"]))
ate,  std_err = get_average_treatment_effect(np.array(groups[list(groups.keys())[0]]["asymptotic_returns"]), np.array(groups[list(groups.keys())[1]]["asymptotic_returns"]))
print("ATE : {:.2f} \pm {:.2f}".format(ate, std_err))
print("There was {}a significance difference with p={:.2f} and threshold .05.".format("" if significant_difference else "not ", p))

# plt.savefig('plot.png')