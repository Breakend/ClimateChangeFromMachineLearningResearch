import atexit
import subprocess
import time

from collections import OrderedDict
from subprocess import PIPE, Popen
from xml.etree.ElementTree import fromstring
from experiment_impact_tracker.utils import *

import numpy as np
import pandas as pd

import psutil

_timer = getattr(time, 'monotonic', time.time)


def _stringify_performance_states(state_dict):
    return "|".join("::".join(map(lambda x: str(x), z)) for z in state_dict.items())


def get_nvidia_gpu_power(pid_list, logger=None):
    # Find per process per gpu usage info
    sp = subprocess.Popen(['nvidia-smi', 'pmon', '-c', '10'],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_str_split = out_str[0].decode('utf-8').split('\n')
    # sometimes with too many processess on the machine or too many gpus, this command will reprint the headers
    # to avoid that we just remove duplicate lines
    out_str_split = list(OrderedDict.fromkeys(out_str_split))
    out_str_pruned = [out_str_split[0], ] + out_str_split[2:]
    out_str_final = "\n".join(out_str_pruned)
    out_str_final = out_str_final.replace("-", "0")
    df = pd.read_csv(pd.compat.StringIO(
        out_str_final[1:]), engine='python', delim_whitespace=True)
    process_percentage_used_gpu = df.groupby(
        ['gpu', 'pid']).mean().reset_index()

    p = Popen(['nvidia-smi', '-q', '-x'], stdout=PIPE)
    outs, errors = p.communicate()
    xml = fromstring(outs)
    num_gpus = int(xml.findall('attached_gpus')[0].text)
    results = []
    power = 0
    per_gpu_absolute_percent_usage = {}
    per_gpu_relative_percent_usage = {}
    absolute_power = 0
    per_gpu_performance_states = {}

    for gpu_id, gpu in enumerate(xml.findall('gpu')):
        gpu_data = {}

        name = gpu.findall('product_name')[0].text
        gpu_data['name'] = name

        # get memory
        memory_usage = gpu.findall('fb_memory_usage')[0]
        total_memory = memory_usage.findall('total')[0].text
        used_memory = memory_usage.findall('used')[0].text
        free_memory = memory_usage.findall('free')[0].text
        gpu_data['memory'] = {
            'total': total_memory,
            'used_memory': used_memory,
            'free_memory': free_memory
        }

        # get utilization
        utilization = gpu.findall('utilization')[0]
        gpu_util = utilization.findall('gpu_util')[0].text
        memory_util = utilization.findall('memory_util')[0].text
        gpu_data['utilization'] = {
            'gpu_util': gpu_util,
            'memory_util': memory_util
        }

        # get power
        power_readings = gpu.findall('power_readings')[0]
        power_draw = power_readings.findall('power_draw')[0].text

        gpu_data['power_readings'] = {
            'power_draw': power_draw
        }
        absolute_power += float(power_draw.replace("W", ""))

        # processes
        processes = gpu.findall('processes')[0]

        infos = []
        # all the info for processes on this particular gpu that we're on
        gpu_based_processes = process_percentage_used_gpu[process_percentage_used_gpu['gpu'] == gpu_id]
        # what's the total absolute SM for this gpu across all accessible processes
        percentage_of_gpu_used_by_all_processes = float(
            gpu_based_processes['sm'].sum())

        for info in processes.findall('process_info'):
            pid = info.findall('pid')[0].text
            process_name = info.findall('process_name')[0].text
            used_memory = info.findall('used_memory')[0].text
            sm_absolute_percent = gpu_based_processes[gpu_based_processes['pid'] == int(
                pid)]['sm'].sum()
            if percentage_of_gpu_used_by_all_processes == 0:
                # avoid divide by zero, sometimes nothing is used so 0/0 should = 0 in this case
                sm_relative_percent = 0
            else:
                sm_relative_percent = sm_absolute_percent / \
                    percentage_of_gpu_used_by_all_processes
            infos.append({
                'pid': pid,
                'process_name': process_name,
                'used_memory': used_memory,
                'sm_relative_percent': sm_relative_percent,
                'sm_absolute_percent': sm_absolute_percent
            })

            if int(pid) in pid_list:
                # only add a gpu to the list if it's being used by one of the processes. sometimes nvidia-smi seems to list all gpus available
                # even if they're not being used by our application, this is a problem in a slurm setting
                if gpu_id not in per_gpu_absolute_percent_usage:
                    # percentage_of_gpu_used_by_all_processes
                    per_gpu_absolute_percent_usage[gpu_id] = 0
                if gpu_id not in per_gpu_relative_percent_usage:
                    # percentage_of_gpu_used_by_all_processes
                    per_gpu_relative_percent_usage[gpu_id] = 0

                if gpu_id not in per_gpu_performance_states:
                    # we only log information for gpus that we're using, we've noticed that nvidia-smi will sometimes return information
                    # about all gpu's on a slurm cluster even if they're not assigned to a worker
                    performance_state = gpu.findall(
                        'performance_state')[0].text
                    per_gpu_performance_states[gpu_id] = performance_state

                power += sm_relative_percent * \
                    float(power_draw.replace("W", ""))
                # want a proportion value rather than percentage
                per_gpu_absolute_percent_usage[gpu_id] += (
                    sm_absolute_percent / 100.0)
                per_gpu_relative_percent_usage[gpu_id] += sm_relative_percent

        gpu_data['processes'] = infos

        results.append(gpu_data)

    average_gpu_utilization = np.mean(
        list(per_gpu_absolute_percent_usage.values()))
    average_gpu_relative_utilization = np.mean(
        list(per_gpu_relative_percent_usage.values()))

    data_return_values_with_headers = {
        "nvidia_draw_absolute": absolute_power,
        "nvidia_estimated_attributable_power_draw": power,
        "average_gpu_estimated_utilization_absolute": average_gpu_utilization,
        "average_gpu_relative_utilization": average_gpu_relative_utilization,
        "per_gpu_performance_state":  _stringify_performance_states(per_gpu_performance_states)
    }

    return data_return_values_with_headers
