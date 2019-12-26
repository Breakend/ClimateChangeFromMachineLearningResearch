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


SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
        # experiment_time, gpu_hours, total_power = np.mean(aggregated_info[exp_set_name]["exp_len_hours"]), np.mean(aggregated_info[exp_set_name]["gpu_hours"]),  np.mean(aggregated_info[exp_set_name]["gpu_hours"])
        for experiment_time, gpu_hours, total_power in zip(aggregated_info[exp_set_name]["exp_len_hours"], aggregated_info[exp_set_name]["gpu_hours"],  aggregated_info[exp_set_name]["gpu_hours"]):
            # j = 
            # experiment_time * num_gpus * .33 * 250 Watts
            
            # gpu-hrs * 250 Watts (TDP)

            # FPOs (?) * Operations/ Watt of GPU
            # 6750920 --> PPO/A2C
            #  7057088 DQN
            # 2.5 MFLOPs * 5M updates * 32 batch size * 2 multiply-add * 3 backward pass / 4 update every 4 steps = 575 Total TFPOs FP32
            # Titan V      14.90 TFLOPS 
            # (2.7e-5/4) PFLOPS-day /(.01490/250) PFLOPS/W)  = W-days / 1000.0 * 24 = kwh
            # [,  , "Ours"]
            l.append((exp_set_name, "Ours", total_power))
            l.append((exp_set_name, "Time x GPUs x 1/3Util x TDP", experiment_time * 1.0 * .33 * 250./1000.))
            l.append((exp_set_name, "Time x GPUs x 1Util x TDP", experiment_time * 1.0 * 1 * 250./1000.))
            l.append((exp_set_name, "GPU-hrsxTDP", gpu_hours * 250.0/1000.0))
            # See Amodei and Hernandez for example of this method in action, since we use the same DQN network as they assume we can use their calculations as a basis for PPO calculations
            # Updates: 2.55M add-multiplies * 5M timesteps / 1024 timesteps before update * 256 minibatch size * 2 multiply-add * 2 backward pass * 4 minibatches per epoch * 4 epochs
            # Sample collection: + 2.55M add-multiplies * 5M timesteps * 2 multiply-add * 1 forward pass 
            # 2550000 * 5000000 * 256 / 1024  * 2  * 2 * 4 *4 + 2550000 * 5000000 *2  
            # = 2.295e+14 / 1000000000000000
            # = 0.2295/86400 PF = 2.96875e-4 pfs-days
            # or 0.00006375PFLOPS-hr / (.01490/250)  PFLOPS/W = 1.06963087248 / 1000 = 0.00106963087 kWh 
            l.append((exp_set_name, "PFLOPs-hr x (GPU PFLOPS/W)", 0.00106963087 )) # PPO

    df = pd.DataFrame(np.array(l), columns=["Experiment", "Estimation Method", "kWh"])
    # df["Experiment"] = df["Experiment"].astype(float)
    for column in df.columns:
        try:
            df[column] = df[column].astype(float)
        except:
            pass
    
    pairs = combinations( df['Estimation Method'].unique(), 2)
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(ax=ax, x="Estimation Method", y="kWh", data=df, capsize=.2)
    add_stat_annotation(ax, data=df, x="Estimation Method", y="kWh",
                    box_pairs= pairs,
                    test='t-test_ind', text_format='star', loc='inside')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=.5)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.savefig('estimation_methods.png', bbox_inches='tight')
    plt.close('all')


    l = []
    sorted_region = get_sorted_region_infos()
    for i, exp_set_name in enumerate(args.experiment_set_names):
        # experiment_time, gpu_hours, total_power = np.mean(aggregated_info[exp_set_name]["exp_len_hours"]), np.mean(aggregated_info[exp_set_name]["gpu_hours"]),  np.mean(aggregated_info[exp_set_name]["gpu_hours"])
        for experiment_time, gpu_hours, total_power, realtime_carbon in zip(aggregated_info[exp_set_name]["exp_len_hours"], aggregated_info[exp_set_name]["gpu_hours"],  aggregated_info[exp_set_name]["gpu_hours"], aggregated_info[exp_set_name]["estimated_carbon_impact_kg"]):
        # j = 
        # experiment_time * num_gpus * .33 * 250 Watts
        
        # gpu-hrs * 250 Watts (TDP)

        # FPOs (?) * Operations/ Watt of GPU
        # 6750920 --> PPO/A2C
        #  7057088 DQN
        # 2.5 MFLOPs * 5M updates * 32 batch size * 2 multiply-add * 3 backward pass / 4 update every 4 steps = 575 Total TFPOs FP32
        # Titan V      14.90 TFLOPS 
        # (2.7e-5/4) PFLOPS-day /(.01490/250) PFLOPS/W)  = W-days / 1000.0 * 24 = kwh
        # [,  , "Ours"]
            l.append((exp_set_name, "Power x Realtime", realtime_carbon))
            l.append((exp_set_name, "Power x CA Average", total_power * 250.73337617853463/1000.))
            l.append((exp_set_name, "Power x EPA US Average", total_power * 432.72712 / 1000.0))

    df = pd.DataFrame(np.array(l), columns=["Experiment", "Estimation Method", "kgCO2eq"])
    # df["Experiment"] = df["Experiment"].astype(float)
    for column in df.columns:
        try:
            df[column] = df[column].astype(float)
        except:
            pass
    
    pairs = combinations( df['Estimation Method'].unique(), 2)    
    a4_dims = (14, 9)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(ax=ax,x="Estimation Method", y="kgCO2eq", data=df, capsize=.2)
    add_stat_annotation(ax, data=df, x="Estimation Method", y="kgCO2eq",
                    box_pairs= pairs,
                    test='t-test_ind', text_format='star', loc='inside')
    plt.margins(0.2)
    plt.subplots_adjust(bottom=.5)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.savefig('estimation_methods_carbon.png', bbox_inches='tight')
    plt.close('all')

    # l = []
    # sorted_region = get_sorted_region_infos()
    # for i, exp_set_name in enumerate(args.experiment_set_names):
    #     for val in aggregated_info[exp_set_name][args.y_axis_var]:
    #         # j = 0
    #         for region_id, inf in sorted_region:
    #             try:
    #                 name = get_zone_name_by_id(region_id)
    #                 if name in WHITELIST_REGION:
    #                     if "45-50 words" not in exp_set_name:
    #                         continue
    #                     exp_superset = "Conv" if "conv" in exp_set_name.lower() else "Transformer"
    #                     l.append((name, exp_superset, val * inf / 1000.))
    #             except:
    #                 continue

    # df = pd.DataFrame(np.array(l), columns=["Region", "Model", "kgCO2"])
    # # df["Experiment"] = df["Experiment"].astype(float)
    # df["kgCO2"] = df["kgCO2"].astype(float)
    # ax = sns.barplot(x="Region", y="kgCO2", hue="Model", data=df, capsize=.2)
    # add_stat_annotation(ax, data=df, x="Region", y="kgCO2", hue="Model",
    #                 box_pairs=[((day, "Conv"), (day, "Transformer")) for day in df['Region'].unique()],
    #                 test='t-test_ind', text_format='star', loc='inside',
    #                 stack=False)
    # plt.margins(0.2)
    # plt.subplots_adjust(bottom=.5)
    # plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # plt.savefig('translation_region.png', bbox_inches='tight')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
