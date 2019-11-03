import seaborn as sns
import pandas as pd
import argparse
import os
import sys
from datetime import datetime
import scipy
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from experiment_impact_tracker.compute_tracker import (PUE,
                                                       load_data_into_frame,
                                                       load_initial_info)
                                                       
from experiment_impact_tracker.stats import run_test, get_average_treatment_effect
from experiment_impact_tracker.emissions.common import get_realtime_carbon_source
from pylatex import (Axis, Document, Figure, LineBreak, Math, Package, Plot,
                     Section, Subsection, Subsubsection, Table, Tabular, TikZ)
from pylatex.utils import escape_latex
import json
from deepdiff import DeepDiff  # For Deep Difference of 2 objects
from itertools import combinations 
# --- Use the 'palette' argument of seaborn

def _convert_from_abbrev_to_numeric(data):
    m = {'K': 3, 'M': 6, 'B': 9, 'T': 12}
    return float(data[:-1]) * 10 ** m[data[-1]] 

def main(arguments):

    parser=argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logdirs', nargs = '+',
                        help = "Input directories", type = str)
    parser.add_argument('--experiment_set_names', nargs="*")
    parser.add_argument('--experiment_set_filters', nargs="*")
    parser.add_argument('--additional_data_per_set', nargs="*")
    parser.add_argument('--y_axis_var', type=str)
    args=parser.parse_args(arguments)
    
    # TODO: add flag for summary stats instead of table for each, this should create a shorter appendix

    aggregated_info = {}

    for exp_set, _filter in enumerate(args.experiment_set_filters):
        aggregated_info[args.experiment_set_names[exp_set]] = {}

        # Allow for a sort of regex filter
        if "*" in _filter:
            _filter = _filter.split("*")
        else:
            _filter = [_filter]
        def check(va):
            for _filt in _filter:
                if _filt not in va:
                    return False
            return True

        filtered_dirs = list(filter(check, args.logdirs))
        print("Filtered dirs: {}".format(",".join(filtered_dirs)))
        for i, x in enumerate(filtered_dirs):
            info = load_initial_info(x)
            extracted_info = gather_additional_info(info, x)
            for key, value in extracted_info.items():
                if key not in aggregated_info[args.experiment_set_names[exp_set]]:
                    aggregated_info[args.experiment_set_names[exp_set]][key] = []
                aggregated_info[args.experiment_set_names[exp_set]][key].append(value)

    l = []
    for i, exp_set_name in enumerate(args.experiment_set_names):
        # for val in :
        val = np.mean(aggregated_info[exp_set_name][args.y_axis_var])
        l.append((exp_set_name, val, _convert_from_abbrev_to_numeric(args.additional_data_per_set[i])))
    df = pd.DataFrame(np.array(l), columns=["exp_set_name", args.y_axis_var, "FPOs"])
    df[args.y_axis_var] = df[args.y_axis_var].astype(float)
    df["FPOs"] = df["FPOs"].astype(float)
    f, ax = plt.subplots(figsize=(7, 14))
    ax.set(xscale="log")

    graph = sns.lmplot(x=args.y_axis_var, y="FPOs", data=df, fit_reg=False, hue='exp_set_name', legend=True, palette="Set1")
    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, scatter=False, ax=graph.axes[2])
    plt.savefig('plot.png')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
