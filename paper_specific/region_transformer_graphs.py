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
from experiment_impact_tracker.data_utils import load_data_into_frame, load_initial_info
from experiment_impact_tracker.utils import gather_additional_info


from experiment_impact_tracker.stats import run_test, get_average_treatment_effect
from experiment_impact_tracker.get_region_metrics import get_current_region_info, get_sorted_region_infos, get_zone_information_by_coords,get_zone_name_by_id
from experiment_impact_tracker.emissions.common import get_realtime_carbon_source
from pylatex import (Axis, Document, Figure, LineBreak, Math, Package, Plot,
                     Section, Subsection, Subsubsection, Table, Tabular, TikZ)
from pylatex.utils import escape_latex
import json
from deepdiff import DeepDiff  # For Deep Difference of 2 objects
from itertools import combinations
# --- Use the 'palette' argument of seaborn

WHITELIST_REGION = {
    'Québec, Canada',
    'Oregon, United States of America',
    'Hong Kong',
    'Alberta, Canada',
    'Tamil Nadu, India',
    'Estonia',
    'Germany',
    'South Brazil, Brazil',
    'Tōkyō, Japan',
    'California, United States of America',
    'New South Wales, Australia',
    'Maharashtra, India',
    'Great Britain',
    ''
}

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
    sorted_region = get_sorted_region_infos()
    for i, exp_set_name in enumerate(args.experiment_set_names):
        for val in aggregated_info[exp_set_name][args.y_axis_var]:
            # j = 0
            for region_id, inf in sorted_region:
                try:
                    name = get_zone_name_by_id(region_id)
                    if name in WHITELIST_REGION:
                        l.append((exp_set_name, val * inf / 1000. , inf))
                except:
                    continue

    df = pd.DataFrame(np.array(l), columns=["exp_set_name", "kg CO2", "intensity"])
    df["kg CO2"] = df["kg CO2"].astype(float)
    df["intensity"] = df["intensity"].astype(float)
    # f, ax = plt.subplots(figsize=(8, 24))
    # ax.set(xscale="log")

    graph = sns.lmplot( x="intensity", y="kg CO2", data=df, fit_reg=True, hue='exp_set_name', legend=True, size = 10, aspect = 1.2)
    ax = graph.axes[0,0]
    labels = [w.get_text() for w in ax.get_xticklabels()]
    locs=list(ax.get_xticks())
    # labels+=[r'$\pi$']
    # locs+=[pi]

    for i, exp_set_name in enumerate(args.experiment_set_names):
        if i==0: continue
        val = np.mean(aggregated_info[exp_set_name][args.y_axis_var])
            # j = 0
        z= 0
        for region_id, inf in sorted_region:
            z += 1
            try:
                name = get_zone_name_by_id(region_id)
            except:
                continue
            print(name)
            if name in WHITELIST_REGION:
            # if not "Canada" in name and not "India" in name:
            #     if z < len(sorted_region) and z%5 != 0 :continue
                locs.append(inf)
                labels.append(name)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xticks(locs)
    ax.set_xlim([0,1100])
    ax.set_ylim([0,0.04])

    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Pad margins so that markers don't get clipped by the axes
    plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=.5)
    # def label_point(x, y, val, ax):
    #     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    #     for i, point in a.iterrows():
    #         ax.text(point['x']+.02, point['y'], str(point['val']))

    # label_point(df_iris.sepal_length, df_iris.sepal_width, df_iris.species, plt.gca())
    # plt.legend(loc='lower right')
    #Use regplot to plot the regression line for the whole points
    # sns.regplot(x="FPOs", y=args.y_axis_var, data=df, scatter=False, ax=graph.axes[2])
    plt.savefig('plot.png')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
