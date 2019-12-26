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
from experiment_impact_tracker.emissions.get_region_metrics import get_current_region_info, get_sorted_region_infos, get_zone_information_by_coords,get_zone_name_by_id
from experiment_impact_tracker.emissions.common import get_realtime_carbon_source
from pylatex import (Axis, Document, Figure, LineBreak, Math, Package, Plot,
                     Section, Subsection, Subsubsection, Table, Tabular, TikZ)
from pylatex.utils import escape_latex
import json
from deepdiff import DeepDiff  # For Deep Difference of 2 objects
from itertools import combinations
# --- Use the 'palette' argument of seaborn
from statannot import add_stat_annotation
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


    all_log_dirs = []



    for log_dir in args.logdirs:
        for path, subdirs, files in os.walk(log_dir):
            if "impacttracker" in path:
                all_log_dirs.append(path.replace("impacttracker", "").replace("//", "/"))

    all_log_dirs = list(set(all_log_dirs))

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

        filtered_dirs = list(filter(check, all_log_dirs))
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
            # j = 
            if "3-50 words" not in exp_set_name:
                continue
            exp_superset = "Conv" if "conv" in exp_set_name.lower() else "Transformer"
            l.append((exp_set_name.replace("Conv","").replace("Transformer","").replace("(","").replace(")",""), exp_superset, val))

    df = pd.DataFrame(np.array(l), columns=["Sentence Length Distribution", "Model", "kWh"])
    # df["Experiment"] = df["Experiment"].astype(float)
    df["kWh"] = df["kWh"].astype(float)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax = sns.barplot(ax=ax, x="Sentence Length Distribution", y="kWh", hue="Model", data=df, capsize=.2)
    add_stat_annotation(ax, data=df, x="Sentence Length Distribution", y="kWh", hue="Model",
                    box_pairs=[((day, "Conv"), (day, "Transformer")) for day in df['Sentence Length Distribution'].unique()],
                    test='t-test_ind', text_format='star', loc='inside',
                    stack=False)
    plt.margins(0.2)
    plt.subplots_adjust(bottom=.5)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.savefig('translation_word_count.png', bbox_inches='tight')
    plt.close('all')

    l = []
    sorted_region = get_sorted_region_infos()
    for i, exp_set_name in enumerate(args.experiment_set_names):
        for val in aggregated_info[exp_set_name][args.y_axis_var]:
            # j = 0
            for region_id, inf in sorted_region:
                try:
                    name = get_zone_name_by_id(region_id)
                    if name in WHITELIST_REGION:
                        if "3-50 words" not in exp_set_name:
                            continue
                        exp_superset = "Conv" if "conv" in exp_set_name.lower() else "Transformer"
                        l.append((name.replace("United States of America", "USA").replace("New South Wales", "NSW"), exp_superset, val * inf / 1000.))
                except:
                    continue

    df = pd.DataFrame(np.array(l), columns=["Region", "Model", "kgCO2"])
    # df["Experiment"] = df["Experiment"].astype(float)
    df["kgCO2"] = df["kgCO2"].astype(float)
    quebec = df[df["Region"] == "Québec, Canada"]
    estonia = (df[df["Region"] == "Estonia"])
    quebec_transformer = quebec[df["Model"] == "Transformer"]
    quebec_conv = quebec[df["Model"] == "Conv"]
    estonia_transformer = estonia[df["Model"] == "Transformer"]
    estonia_conv = estonia[df["Model"] == "Conv"]

    estonia_diff, estonia_scale = (np.mean(estonia_conv["kgCO2"]) - np.mean(estonia_transformer["kgCO2"]), np.mean(estonia_conv["kgCO2"])/np.mean(estonia_transformer["kgCO2"]))
    quebec_diff, quebec_scale = (np.mean(quebec_conv["kgCO2"]) - np.mean(quebec_transformer["kgCO2"]), np.mean(quebec_conv["kgCO2"])/np.mean(quebec_transformer["kgCO2"]))
    fig, ax = plt.subplots(figsize=(6, 8))
    ax = sns.barplot(ax=ax, x="Region", y="kgCO2", hue="Model", data=df, capsize=.2)
    add_stat_annotation(ax, data=df, x="Region", y="kgCO2", hue="Model",
                    box_pairs=[((day, "Conv"), (day, "Transformer")) for day in df['Region'].unique()],
                    test='t-test_ind', text_format='star', loc='inside',
                    stack=False)
    ax.text(13.2, .026, "Diff:\n{:.2f} gCO2eq\n(~x{:.2f})".format(estonia_diff*1000, estonia_scale),
        horizontalalignment="center", verticalalignment="top", fontsize=10,
        backgroundcolor=(1., 1., 1., .3)) 
    ax.text(-1.1, .012, "Diff:\n{:.2f} gCO2eq\n(~x{:.2f})".format(quebec_diff*1000, quebec_scale),
        horizontalalignment="center", verticalalignment="top", fontsize=10,
        backgroundcolor=(1., 1., 1., .3))
    plt.margins(0.2)
    plt.subplots_adjust(bottom=.5)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.savefig('translation_region.png', bbox_inches='tight')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
