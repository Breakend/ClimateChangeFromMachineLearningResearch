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
            for key, value in extracted_info.items():
                if key not in aggregated_info[args.experiment_set_names[exp_set]]:
                    aggregated_info[args.experiment_set_names[exp_set]][key] = []
                aggregated_info[args.experiment_set_names[exp_set]][key].append(value)

    flop_thing = pd.read_csv('/Users/breakend/Documents/code/machine_learning/ClimateChangeFromMachineLearningResearch/paper_specific/flops_imagenet.txt', sep='|\s+', encoding="utf-8", skipinitialspace=True, delimiter="|")
    l = []
    for i, exp_set_name in enumerate(args.experiment_set_names):
        # for val in :
        val = np.mean(aggregated_info[exp_set_name]["total_power"])
        exp_set_name_val = exp_set_name
        if args.fit:
            if "google" in exp_set_name or "alex" in exp_set_name or "squeeze" in exp_set_name or "hard" in exp_set_name:
                continue 
            exp_set_name_val = exp_set_name.split("_")[0]
            exp_set_name_val = exp_set_name_val.split("1")[0]
            exp_set_name_val = exp_set_name_val.split("5")[0]
            exp_set_name_val = exp_set_name_val.split("8")[0]
            exp_set_name_val = exp_set_name_val.split("2")[0]
            exp_set_name_val = exp_set_name_val.split("3")[0]
            exp_set_name_val = exp_set_name_val.split("6")[0]

        l.append((exp_set_name_val, val, float(flop_thing[flop_thing["Model"] == exp_set_name]["Top-1"]), float(flop_thing[flop_thing["Model"] == exp_set_name]["FLOPs(G)"])))
    df = pd.DataFrame(np.array(l), columns=["Model", "kWh", "Top-1 Accuracy", "FPOs(G)"])
    df["kWh"] = df["kWh"].astype(float)
    df["FPOs(G)"] = df["FPOs(G)"].astype(float)
    df["Top-1 Accuracy"] = df["Top-1 Accuracy"].astype(float)

    from scipy.stats import linregress

    print("FPOs - kwH")
    print(linregress(df["FPOs(G)"], df["kWh"]))

    corr, _ = pearsonr(df["FPOs(G)"], df["kWh"])
    print('Pearsons correlation: %.3f' % corr)

    
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    print(df)
    graph = sns.scatterplot(ax=ax, y="kWh", x="FPOs(G)", data=df, sizes=(250, 500),  alpha=.5, hue='Model', size="Top-1 Accuracy", legend='brief')#, palette="Set1")
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    ax.axhline(0.041845, 0, 4.39/25.0, ls='--')
    ax.axhline(0.040427, 0, 23.84/25.0, ls='--')
    ax.axvline(3.39, 0, 0.041845/0.077, ls='--')
    ax.axvline(22.84, 0, 0.040427/0.077, ls='--') 
    plt.annotate('', (3.39, 0.005), (22.84, 0.005), arrowprops={'arrowstyle': '<->'})
    plt.annotate(
        'Difference: 19.45 GFPOs (~6.64x)', xy=(11.45, 0.005), xycoords='data',
        xytext=(0, 0.5), textcoords='offset points')
    plt.annotate('',  (2.5, 0.040427), (2.5, 0.041845), arrowprops={'arrowstyle': '<->'})
    plt.annotate(
        'Difference: .001 kWh (~1.02x)', xy=(1.5, 0.045), xycoords='data',
         xytext=(0, 0.00), textcoords='offset points')
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('flops_power.png',bbox_inches='tight')
    
    # plot 
    l = []
    for i, exp_set_name in enumerate(args.experiment_set_names):
        # for val in :
        val = np.mean(aggregated_info[exp_set_name]["exp_len_hours"])
        exp_set_name_val = exp_set_name
        if args.fit:
            if "google" in exp_set_name or "alex" in exp_set_name or "squeeze" in exp_set_name or "hard" in exp_set_name:
                continue 
            exp_set_name_val = exp_set_name.split("_")[0]
            exp_set_name_val = exp_set_name_val.split("1")[0]
            exp_set_name_val = exp_set_name_val.split("5")[0]
            exp_set_name_val = exp_set_name_val.split("8")[0]
            exp_set_name_val = exp_set_name_val.split("2")[0]
            exp_set_name_val = exp_set_name_val.split("3")[0]
            exp_set_name_val = exp_set_name_val.split("6")[0]

        l.append((exp_set_name_val, val, float(flop_thing[flop_thing["Model"] == exp_set_name]["Top-1"]), float(flop_thing[flop_thing["Model"] == exp_set_name]["FLOPs(G)"])))
    df = pd.DataFrame(np.array(l), columns=["Model", "Time(h)", "Top-1 Accuracy", "FPOs(G)"])
    df["Time(h)"] = df["Time(h)"].astype(float)
    df["FPOs(G)"] = df["FPOs(G)"].astype(float)
    df["Top-1 Accuracy"] = df["Top-1 Accuracy"].astype(float)
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    graph = sns.scatterplot(ax=ax,y="Time(h)", x="FPOs(G)", data=df, sizes=(250, 500),  alpha=.5, hue='Model', size="Top-1 Accuracy", legend='brief')#, palette="Set1")
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    
    print("FPOs - Time")
    print(linregress(df["FPOs(G)"], df["Time(h)"]))


    corr, _ = pearsonr(df["FPOs(G)"], df["Time(h)"])
    print('Pearsons correlation: %.3f' % corr) 
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('flops_time.png',bbox_inches='tight')

    # plot 
    l = []
    for i, exp_set_name in enumerate(args.experiment_set_names):
        # for val in :
        exp_set_name_val = exp_set_name
        if args.fit:
            if "google" in exp_set_name or "alex" in exp_set_name or "squeeze" in exp_set_name or "hard" in exp_set_name:
                continue 
            exp_set_name_val = exp_set_name.split("_")[0]
            exp_set_name_val = exp_set_name_val.split("1")[0]
            exp_set_name_val = exp_set_name_val.split("5")[0]
            exp_set_name_val = exp_set_name_val.split("8")[0]
            exp_set_name_val = exp_set_name_val.split("2")[0]
            exp_set_name_val = exp_set_name_val.split("3")[0]
            exp_set_name_val = exp_set_name_val.split("6")[0]

        val = np.mean(aggregated_info[exp_set_name]["exp_len_hours"])
        l.append((exp_set_name_val, val, float(flop_thing[flop_thing["Model"] == exp_set_name]["Top-1"]), float(flop_thing[flop_thing["Model"] == exp_set_name]["Params(M)"])))
    df = pd.DataFrame(np.array(l), columns=["Model", "Time(h)", "Top-1 Accuracy", "Params(M)"])
    df["Time(h)"] = df["Time(h)"].astype(float)
    df["Top-1 Accuracy"] = df["Top-1 Accuracy"].astype(float)
    df["Params(M)"] = df["Params(M)"].astype(float)
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    graph = sns.scatterplot(ax=ax, y="Time(h)", x="Params(M)", data=df, sizes=(250, 500),  alpha=.5, hue='Model', size="Top-1 Accuracy", legend='brief')#, palette="Set1")
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    print("Params - Time")
    print(linregress(df["Params(M)"], df["Time(h)"]))

    corr, _ = pearsonr(df["Params(M)"], df["Time(h)"])
    print('Pearsons correlation: %.3f' % corr) 
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('params_time.png',bbox_inches='tight')

    # plot 
    l = []
    for i, exp_set_name in enumerate(args.experiment_set_names):
        # for val in :
        exp_set_name_val = exp_set_name
        if args.fit:
            if "google" in exp_set_name or "alex" in exp_set_name or "squeeze" in exp_set_name or "hard" in exp_set_name:
                continue 
            exp_set_name_val = exp_set_name.split("_")[0]
            exp_set_name_val = exp_set_name_val.split("1")[0]
            exp_set_name_val = exp_set_name_val.split("5")[0]
            exp_set_name_val = exp_set_name_val.split("8")[0]
            exp_set_name_val = exp_set_name_val.split("2")[0]
            exp_set_name_val = exp_set_name_val.split("3")[0]
            exp_set_name_val = exp_set_name_val.split("6")[0]

        val = np.mean(aggregated_info[exp_set_name]["total_power"])
        l.append((exp_set_name_val, val, float(flop_thing[flop_thing["Model"] == exp_set_name]["Top-1"]), float(flop_thing[flop_thing["Model"] == exp_set_name]["Params(M)"])))
    df = pd.DataFrame(np.array(l), columns=["Model", "kWh", "Top-1 Accuracy", "Params(M)"])
    df["kWh"] = df["kWh"].astype(float)
    df["Params(M)"] = df["Params(M)"].astype(float)
    df["Top-1 Accuracy"] = df["Top-1 Accuracy"].astype(float)
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    graph = sns.scatterplot(ax=ax, y="kWh", x="Params(M)", data=df, sizes=(250, 500),  alpha=.5, hue='Model', size="Top-1 Accuracy", legend='brief')
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.83,box.height])
    print("Params - kwH")
    print(linregress(df["Params(M)"], df["kWh"]))

    corr, _ = pearsonr(df["Params(M)"], df["kWh"])
    print('Pearsons correlation: %.3f' % corr) 
    plt.legend(loc='upper left',bbox_to_anchor=(1,1.15))
    plt.ylim(bottom=0.0)

    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, sizes=(250, 500),  alpha=.5, scatter=False, ax=graph.axes[2])
    plt.savefig('params_power.png',bbox_inches='tight')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
