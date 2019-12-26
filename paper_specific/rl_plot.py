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
from experiment_impact_tracker.data_utils import (load_data_into_frame,
                                                       load_initial_info)
                                                       
from experiment_impact_tracker.stats import run_test, get_average_treatment_effect
from experiment_impact_tracker.utils import gather_additional_info
from experiment_impact_tracker.emissions.common import get_realtime_carbon_source
from pylatex import (Axis, Document, Figure, LineBreak, Math, Package, Plot,
                     Section, Subsection, Subsubsection, Table, Tabular, TikZ)
from pylatex.utils import escape_latex
import json
from deepdiff import DeepDiff  # For Deep Difference of 2 objects
from itertools import combinations 
# --- Use the 'palette' argument of seaborn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    parser.add_argument('--fit', action="store_true", default=False)
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
            rl_eval = pd.read_csv(os.path.join(x, 'eval_test.csv'), skiprows=1)
            if "AverageReturn" not in aggregated_info[args.experiment_set_names[exp_set]]:
                aggregated_info[args.experiment_set_names[exp_set]]["AverageReturn"] = [np.mean(rl_eval["rmean"])]
            else:
                aggregated_info[args.experiment_set_names[exp_set]]["AverageReturn"].append(np.mean(rl_eval["rmean"]))
            if "AsymptoticReturn" not in aggregated_info[args.experiment_set_names[exp_set]]:
                aggregated_info[args.experiment_set_names[exp_set]]["AsymptoticReturn"] = [rl_eval["rmean"][len(rl_eval["rmean"])-1]]
            else:
                aggregated_info[args.experiment_set_names[exp_set]]["AsymptoticReturn"].append(rl_eval["rmean"][len(rl_eval["rmean"])-1])
            for key, value in extracted_info.items():
                if key not in aggregated_info[args.experiment_set_names[exp_set]]:
                    aggregated_info[args.experiment_set_names[exp_set]][key] = []
                aggregated_info[args.experiment_set_names[exp_set]][key].append(value)

    l = []
    for i, exp_set_name in enumerate(args.experiment_set_names):
        for j, val in enumerate(aggregated_info[exp_set_name]["total_power"]):
            exp_set_name_val = exp_set_name
            l.append((exp_set_name_val, val, aggregated_info[exp_set_name]["exp_len_hours"][j], aggregated_info[exp_set_name]["AverageReturn"][j],  aggregated_info[exp_set_name]["AsymptoticReturn"][j]  ))
    df = pd.DataFrame(np.array(l), columns=["Algo", "kWh", "Time(h)", "AverageReturn","AsymptoticReturn"])
    df["Time(h)"] = df["Time(h)"].astype(float)
    df["kWh"] = df["kWh"].astype(float)
    df["AverageReturn"] = df["AverageReturn"].astype(float)
    df["AsymptoticReturn"] = df["AsymptoticReturn"].astype(float)

    # from scipy.stats import linregress

    # print("FPOs - kwH")
    # print(linregress(df["FPOs(G)"], df["kWh"]))

    # corr, _ = pearsonr(df["FPOs(G)"], df["kWh"])
    # print('Pearsons correlation: %.3f' % corr)

    
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    print(df)
    graph = sns.scatterplot(ax=ax, x="kWh", y="AverageReturn", data=df, sizes=(250, 500),  alpha=.5, hue='Algo', size=650, legend='brief', palette=sns.color_palette("colorblind", 10))#, palette="Set1")
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    # plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('kw_averagereturn.png',bbox_inches='tight')

    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    print(df)
    graph = sns.scatterplot(ax=ax, x="kWh", y="AsymptoticReturn", data=df, sizes=(250, 500),  alpha=.5, hue='Algo', size=650,  legend='brief', palette=sns.color_palette("colorblind", 10))#, palette="Set1")
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    # plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('kw_asymptoticreturn.png',bbox_inches='tight')


    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    print(df)
    graph = sns.scatterplot(ax=ax, x="Time(h)", y="AsymptoticReturn", data=df, sizes=(250, 500),  alpha=.5, hue='Algo', size=650,  legend='brief', palette=sns.color_palette("colorblind", 10))#, palette="Set1")
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    # plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('time_asymptoticreturn.png',bbox_inches='tight')
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    print(df)
    graph = sns.scatterplot(ax=ax, x="Time(h)", y="AverageReturn", data=df, sizes=(250, 500),  alpha=.5, hue='Algo', size=650, legend='brief', palette=sns.color_palette("colorblind", 10))#, palette="Set1")
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    # plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('time_averagereturn.png', bbox_inches='tight')

    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
