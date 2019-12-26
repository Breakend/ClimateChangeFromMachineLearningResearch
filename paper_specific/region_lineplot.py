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

import random
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
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

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
    parser.add_argument('--y_axis_var', type=str)
    args=parser.parse_args(arguments)

  

    l = []
    labels=[]
    locs = []
    sorted_region = get_sorted_region_infos()


    for region_id, inf in sorted_region:
        try:
            name = get_zone_name_by_id(region_id)
        except:
            continue
        print(name)
        if name in WHITELIST_REGION:
        # if not "Canada" in name and not "India" in name:
        #     if z < len(sorted_region) and z%5 != 0 :continue
            name = name.replace("United States of America", "USA")
            name = name.replace("New South Wales", "NSW")
            locs.append(inf)
            labels.append(name)

    additional_energy = [(24, "Hydropower"), (740, "Biomass"), (820, "Coal"), (1140, "Oil Shale")]

    energy_names = [x[1] for x in additional_energy]

    additional_clouds = [(30, "ca-central-1 (AWS)\nnorthamerica-northeast1 (GCP)\ncanadaeast (Azure)"), (250.73337617853463, "us-west-1 (AWS)\nus-west2 (GCP)\nwestus (Azure)"), (396.93403342452757,"us-east-1 (AWS)\nus-east4 (GCP)\neastus (Azure)"), (492, "ap-northeast-2 (AWS)\nkoreacentral (Azure)"), (970,"ap-south-1 (AWS)\nasia-south1 (GCP)\ncentralindia (Azure)")]
    cloud_names = [x[1] for x in additional_clouds]

    for inf, name in additional_energy:
        locs.append(inf)
        labels.append(name)
    for inf, name in additional_clouds:
        locs.append(inf)
        labels.append(name)

    zipped_list = zip(labels, locs)
    zipped_list = sorted(zipped_list, key=lambda x: x[1])

    levels = np.array([-4, 4, -3, 3, -2, 2, -1, 1, -.5, .5])
    fig, ax = plt.subplots(figsize=(22, 6))

    # Create the base line
    start = min(locs)
    stop = max(locs)
    ax.plot((0, 1200), (0, 0), 'k', alpha=.5)

    # Iterate through releases annotating each one
    for ii, (iname, idate) in enumerate(zipped_list):
        level = levels[ii % len(levels)]
        if "Shale" in iname:
            level = -2
        if "Coal" in iname:
            level = .5
        if "Britain" in iname or "korea" in iname or "ap-south" in iname:
            level = 2.5
        if "Alberta" in iname:
            level = 1.25
        if "ca-central" in iname or "Japan" in iname:
            level = -1.95
        if "Estonia" in iname:
            level = -3.5
        if "us-west" in iname:
            level = -.5
        if "Oregon" in iname:
            level = -1
        if "Hydro" in iname or "Estonia" in iname:
            level = -3.25

        vert = 'top' if level < 0 else 'bottom'

        ax.scatter(idate, 0, s=100, facecolor='w', edgecolor='k', zorder=9999)
        # Plot a line up to the text
        ax.plot((idate, idate), (0, level), c='r', alpha=.7)
        assi=  "center"

        if "Tamil" in iname or "Cali":
            assi = "right"
        if "Maharas" in iname:
            assi = "right"
        if "us-east-1" in iname:
            assi = "center"
        if "ca-central" in iname or "Brazil" in iname or "Britain" in iname:
            assi = "left"
        if "Hydro" in iname or "us-west" in iname:
            assi = "center"
        if "Oregon" in iname or "Germany" in iname or "Estonia" in iname or "Oil Shale" in iname or "koreacentra" in iname or "Biomass" in iname:
            assi = "center"
        if "Alberta" in iname:
            assi = "left"
        if "Hong" in iname or "Qu" in iname:
            assi = "left"
        if "centralindia" in iname:
            assi = "left"
        if "Australia" in iname:
            assi = "center"
        # Give the text a faint background and align it properly
        if iname in energy_names:
            ax.text(idate, level + .1 if level >0 else level - .1, iname,
                horizontalalignment=assi, verticalalignment=vert, fontsize=14,
                backgroundcolor=(1., 1., 1., .3), bbox=dict(facecolor='none', edgecolor='orange', boxstyle='sawtooth,pad=.2'))            
        elif iname in cloud_names:
            ax.text(idate, level + .1 if level >0 else level - .1, iname,
                horizontalalignment=assi, verticalalignment=vert, fontsize=14,
                backgroundcolor=(1., 1., 1., .3), bbox=dict(facecolor='none', edgecolor='red', boxstyle='roundtooth,pad=.2'))
        else:
            ax.text(idate, level, iname,
                horizontalalignment=assi, verticalalignment=vert, fontsize=14,
                backgroundcolor=(1., 1., 1., .3))
    plt.xlabel("g CO2eq/kWh")
    # Set the xticks formatting
    # format xaxis with 3 month intervals
    # ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=3))
    # ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%b %Y"))
    # fig.autofmt_xdate()

    # Remove components for a cleaner look
    plt.setp((ax.get_yticklabels() + ax.get_yticklines() +
            list(ax.spines.values())), visible=False)
    plt.savefig('regions_lineplot.png', bbox_inches='tight')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
