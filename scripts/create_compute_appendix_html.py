#!/usr/bin/env python3

from __future__ import print_function

import argparse
import os
import sys
from datetime import datetime
import scipy
import re
import pandas as pd
import numpy as np
from pprint import pprint
from jinja2 import Environment, FileSystemLoader
import experiment_impact_tracker
from experiment_impact_tracker.data_utils import load_data_into_frame, load_initial_info, zip_data_and_info
from experiment_impact_tracker.constants import PUE         
from experiment_impact_tracker.stats import run_test, get_average_treatment_effect
from experiment_impact_tracker.emissions.common import get_realtime_carbon_source
from experiment_impact_tracker.get_region_metrics import get_zone_name_by_id
from shutil import copyfile
from experiment_impact_tracker.create_graph_appendix import create_graphs
import json
from deepdiff import DeepDiff  # For Deep Difference of 2 objects
from itertools import combinations 
# def _easy_compare(dict_1, dict_2):
#     dump = json.dumps(dict_1, sort_keys=True)
#     dump2 = json.dumps(dict_2, sort_keys=True)
#     return dump == dump2 

pd.set_option('display.max_colwidth', -1)
# TODO: give experiment set a name and then iterate through each one
# TODO: Each individual one should make a new file and then a summary file and a differences file

from experiment_impact_tracker.utils import gather_additional_info


def _format_setname(setname):
    return setname.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _get_carbon_infos(info, extended_info):

    vals = {}
    if "average_realtime_carbon_intensity" in extended_info:
        vals['Realtime Carbon Intensity Data Source'] = [get_realtime_carbon_source(info["region"]["id"])]
        vals['Realtime Carbon Intensity Average During Exp'] = [extended_info["average_realtime_carbon_intensity"]]
    
    vals["Region Average Carbon Intensity"] = [info["region_carbon_intensity_estimate"]["carbonIntensity"]]
    vals["Region Average Carbon Intensity Source"] = [info["region_carbon_intensity_estimate"]["_source"]]
    vals["Assumed PUE"] = [PUE] 
    vals["Compute Region"] = [get_zone_name_by_id(info["region"]["id"])]
    vals["Experiment Impact Tracker Version"] = [info["experiment_impact_tracker_version"]]
    return pd.DataFrame.from_dict(vals)


def _construct_index_page(output_directory, 
                          aggregated_info, 
                          experiment_set_names, 
                          experiment_set_filters, 
                          executive_summary_variables,
                          description,
                          title,
                          base_dir):
    os.makedirs(output_directory, exist_ok=True)
    template_directory = os.path.join(os.path.dirname(experiment_impact_tracker.__file__), 'html_templates')
    file_loader = FileSystemLoader(template_directory)
    env = Environment(loader=file_loader)

    template = env.get_template('index.html')

    # Gather variables to generate executive summary for
    executive_summary = [["Experiment"] + executive_summary_variables]
    for exp_name in experiment_set_names:
        data = [exp_name]
        for variable in executive_summary_variables:
            values = aggregated_info[exp_name][variable]
            values_mean = np.mean(values)
            values_stdder = scipy.stats.sem(values)
            data.append("{:.3f} +/- {:.2f}".format(values_mean, values_stdder))
        executive_summary.append(data)
    executive_summary = pd.DataFrame(np.vstack(executive_summary))

    output = template.render(
        exp_set_names_titles = [(_format_setname(experiment_set_names[exp_set]), experiment_set_names[exp_set]) for exp_set in range(len(experiment_set_filters))],
        executive_summary = executive_summary,
        title=title,
        description=description,
        relative_base_dir=os.path.relpath(base_dir, output_directory)
        )

    with open(os.path.join(output_directory, 'index.html'), 'w') as f:
        f.write(output)

def _filter_dirs(all_log_dirs, _filter):
    if _filter is None:
        return all_log_dirs
    # Allow for a sort of regex filter
    check = lambda x : bool(re.search(_filter, x)) 
    filtered_dirs = list(filter(check, all_log_dirs))
    print("Filtered dirs: {}".format(",".join(filtered_dirs)))
    return filtered_dirs
    

def _aggregated_data_for_filterset(output_dir, all_dirs, experiment_set_names, experiment_set_filters, only_summary_level=True):
    aggregated_info = {}

    gpu_infos_all = {} 
    cpu_infos_all = {} 
    carbon_infos_all = {}
    package_infos_all = {}
    graph_paths_all = {}
    data_zip_paths_all = {}
    for exp_set, _filter in enumerate(experiment_set_filters):
        aggregated_info[experiment_set_names[exp_set]] = {}

        gpu_infos_all[experiment_set_names[exp_set]] = []
        cpu_infos_all[experiment_set_names[exp_set]] = []
        carbon_infos_all[experiment_set_names[exp_set]] = []
        package_infos_all[experiment_set_names[exp_set]] = []
        graph_paths_all[experiment_set_names[exp_set]] = []
        data_zip_paths_all[experiment_set_names[exp_set]] = []
        filtered_dirs = _filter_dirs(all_dirs, _filter)
        
        for i, x in enumerate(filtered_dirs):
            info = load_initial_info(x)
            extracted_info = gather_additional_info(info, x)
            for key, value in extracted_info.items():
                if key not in aggregated_info[experiment_set_names[exp_set]]:
                    aggregated_info[experiment_set_names[exp_set]][key] = []
                aggregated_info[experiment_set_names[exp_set]][key].append(value)


            if not only_summary_level:
                # create graphs and add it to the experiment set for import to the html page later
                graph_dir = os.path.join(output_dir, _format_setname(experiment_set_names[exp_set]), 'images/')
                graph_paths = create_graphs(x, output_path=graph_dir, max_level=1)
                graph_paths_all[experiment_set_names[exp_set]].append(graph_paths)  

                data_zip_path = os.path.join(output_dir, _format_setname(experiment_set_names[exp_set]), "data")
                os.makedirs(data_zip_path, exist_ok=True)
                # Zip the raw data
                zip_file_name = os.path.join(data_zip_path, "{}.zip".format(i))
                data_zip_paths_all[experiment_set_names[exp_set]].append(zip_file_name)  
                zip_data_and_info(x, zip_file_name)

                gpu_data_frame = pd.DataFrame.from_dict( info["gpu_info"])#{k: [v] for k, v in info["gpu_info"].items()})   
                gpu_infos_all[experiment_set_names[exp_set]].append(gpu_data_frame)
                cpu_data_frame = pd.DataFrame.from_dict({k: [v] for k, v in info["cpu_info"].items()})
                cpu_infos_all[experiment_set_names[exp_set]].append(cpu_data_frame)
                carbon_infos_all[experiment_set_names[exp_set]].append(_get_carbon_infos(info, extracted_info))
                package_infos_all[experiment_set_names[exp_set]].append(info["python_package_info"])
    return {
        "aggregated_info" : aggregated_info,
        "cpu_infos_all" : cpu_infos_all,
        "gpu_infos_all" : gpu_infos_all,
        "carbon_infos_all" : carbon_infos_all,
        "data_zip_paths_all" : data_zip_paths_all,
        "graph_paths_all" : graph_paths_all,
        "package_infos_all" : package_infos_all
    }


def _create_leaf_page(output_directory, all_infos, exp_set_name, experiment_set_names, experiment_set_filters, base_dir):
    template_directory = os.path.join(os.path.dirname(experiment_impact_tracker.__file__), 'html_templates')
    file_loader = FileSystemLoader(template_directory)
    env = Environment(loader=file_loader)
    template = env.get_template('exp_set_index.html')
    aggregated_info, cpu_infos_all, gpu_infos_all, carbon_infos_all, data_zip_paths_all, graph_paths_all, package_infos_all = list(all_infos.values())
    summary_info = [["Value", "Mean", "StdErr", "Sum"]]

    for key, values in aggregated_info[exp_set_name].items():
        values_mean = np.mean(values)
        values_stdder = scipy.stats.sem(values)
        values_summed = np.sum(values)
        summary_info.append((key, values_mean, values_stdder, values_summed))

    output = template.render(
                                exp_set_names_titles = [(_format_setname(experiment_set_names[exp_set]), experiment_set_names[exp_set]) for exp_set in range(len(experiment_set_filters))],
                                exps = list(range(len(list(aggregated_info[exp_set_name].values())[0]))),
                                summary = pd.DataFrame(summary_info),
                                title=exp_set_name,
                                relative_base_dir=os.path.relpath(base_dir, output_directory)
                                )
    os.makedirs(os.path.join(output_directory, _format_setname(exp_set_name)), exist_ok=True)
    with open(os.path.join(output_directory, _format_setname(exp_set_name), 'index.html'), 'w') as f:
        f.write(output)

    
    for i in range(len(list(aggregated_info[exp_set_name].values())[0])):
        
        summary_info = [["Key", "Value"]]

        for key, values in aggregated_info[exp_set_name].items():
            summary_info.append((key, values[i]))


        template = env.get_template('exp_details.html')
        html_output_dir = os.path.join(output_directory, _format_setname(exp_set_name))
        html_output_path =  os.path.join(html_output_dir, '{}.html'.format(i))
        relative_graph_paths = [os.path.relpath(graph_path, html_output_dir) for graph_path in graph_paths_all[exp_set_name][i]]
        relative_data_zip_paths = os.path.relpath(data_zip_paths_all[exp_set_name][i], html_output_dir)
        output = template.render(
                        exp_set_names_titles = [(_format_setname(experiment_set_names[exp_set]), experiment_set_names[exp_set]) for exp_set in range(len(experiment_set_filters))],
                        exps = list(range(len(list(aggregated_info[exp_set_name].values())[0]))),
                        cpu_info = cpu_infos_all[exp_set_name][i].T,
                        gpu_info = gpu_infos_all[exp_set_name][i].T,
                        carbon_info = carbon_infos_all[exp_set_name][i].T,
                        package  = pd.DataFrame.from_dict(package_infos_all[exp_set_name][i]),
                        stats = pd.DataFrame(summary_info),
                        graph_paths=relative_graph_paths,
                        data_download_path = relative_data_zip_paths,
                        title=exp_set_name,
                        relative_base_dir=os.path.relpath(base_dir, output_directory))
        with open(html_output_path, 'w') as f:
            f.write(output)



def _recursive_create(all_log_dirs, output_directory, experiment_def, base_dir=None):

    if base_dir is None:
        base_dir = output_directory

    for experiment_set, values in experiment_def.items():
        filtered_dirs = _filter_dirs(all_log_dirs, values["filter"])

        if "child_experiments" in values:
            # get the names of the child experiments from the keys
            experiment_set_names = list(values["child_experiments"].keys())
            # get the filters from each of them
            experiment_set_filters = [x["filter"] for x in values["child_experiments"].values()]
            # get top level info only since this isn't a leaf node
            
            aggregated_info = _aggregated_data_for_filterset(output_directory, filtered_dirs, experiment_set_names, experiment_set_filters, only_summary_level=True)["aggregated_info"]
            # what should we use to summarize this
            executive_summary_variables = values["executive_summary_variables"]

            new_output_dir = os.path.join(output_directory, _format_setname(experiment_set))

            # Construct the index page
            _construct_index_page(new_output_dir, 
                            aggregated_info, 
                            experiment_set_names, 
                            experiment_set_filters, 
                            executive_summary_variables,
                            values["description"],
                            experiment_set,
                            base_dir=base_dir)
            _recursive_create(filtered_dirs, new_output_dir, values["child_experiments"], base_dir=base_dir)

        else:
            experiment_set_names = list(experiment_def.keys())
            # get the filters from each of them
            experiment_set_filters = [x["filter"] for x in experiment_def.values()]
            # if we're at a leaf experiment set, this is the final bit of aggregation and we show off individual experiments in the set
            all_infos = _aggregated_data_for_filterset(output_directory, filtered_dirs, experiment_set_names, experiment_set_filters, only_summary_level=False)
            
            _create_leaf_page(output_directory, all_infos, experiment_set, experiment_set_names, experiment_set_filters, base_dir=base_dir)



def main(arguments):

    parser=argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('logdirs', nargs = '+',
                        help = "Input directories", type = str)
    parser.add_argument("--title", type=str, default="Experiment Set Information")
    parser.add_argument("--description", type=str, default="TODO: description of experimental setups")
    parser.add_argument("--site_spec", type=str, required=True)
    parser.add_argument('--output_dir')
    args=parser.parse_args(arguments)
    
    # TODO: add flag for summary stats instead of table for each, this should create a shorter appendix
    with open(args.site_spec, 'r') as f:
        site_spec = json.load(f)



    all_log_dirs = []



    for log_dir in args.logdirs:
        for path, subdirs, files in os.walk(log_dir):
            if "impacttracker" in path:
                all_log_dirs.append(path.replace("impacttracker", "").replace("//", "/"))

    all_log_dirs = list(set(all_log_dirs))

    # Create html directory with index from Jinja template


    # copy CSS files
    output_style_dir = os.path.join(args.output_dir, "style/")
    os.makedirs(output_style_dir, exist_ok=True)
    template_directory = os.path.join(os.path.dirname(experiment_impact_tracker.__file__), 'html_templates')
    for root, dirs, files in os.walk(os.path.join(template_directory, "style")):
        for f in files:
            copyfile(os.path.join(root, f), os.path.join(output_style_dir, f))


    _recursive_create(all_log_dirs, args.output_dir, site_spec)  


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
