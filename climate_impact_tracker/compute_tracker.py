import psutil,os
from climate_impact_tracker import rapl
import time
import subprocess
import pandas as pd
import threading

from datetime import datetime

from subprocess import Popen, PIPE
from xml.etree.ElementTree import fromstring

import os
import sys
import traceback
from functools import wraps
from .processor_info import get_my_cpu_info, get_gpu_info
from multiprocessing import Process, Queue
import csv

DATAPATH = 'impacttracker/data.csv'
INFOPATH = 'impacttracker/info.pkl'
SLEEP_TIME = 60
PUE = 1.58 

def safe_file_path(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

def write_csv_data_to_file(file_path, data):
    file_path = safe_file_path(file_path)
    with open(file_path, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (str(ex_value), tb_str)
            raise ex_type(message)

        return ret
    return wrapper


def get_rapl_power(pid_list):
        s1 = rapl.RAPLMonitor.sample()
        time.sleep(5)
        s2 = rapl.RAPLMonitor.sample()
        diff = s2 - s1
        total_intel_power = 0
        total_dram_power = 0
        total_cpu_power = 0
        total_gpu_power = 0
        for d in diff.domains:
            domain = diff.domains[d]
            power = diff.average_power(package=domain.name)
            total_intel_power += power
            for sd in domain.subdomains:
                subdomain = domain.subdomains[sd]
                power = diff.average_power(package=domain.name, domain=subdomain.name)
                subdomain = subdomain.lower()
                if subdomain == "ram" or subdomain == "dram":
                    total_dram_power += power
                elif subdomain == "cores" or subdomain == "cpu":
                    total_cpu_power += power
                elif subdomain == "gpu":
                    total_gpu_power += power

        if total_gpu_power != 0:
            raise ValueError("Don't support credit assignment to Intel RAPL GPU yet.")

        cpu_percent = 0
        mem_percent = 0
        cpu_times = 0
        for process in pid_list:
            p = psutil.Process(process)
            cpu_percent += p.cpu_percent()
            cpu_times += p.cpu_times().user + p.cpu_times().system
            mem_percent += p.memory_percent()

        system_wide_cpu_percent = psutil.cpu_percent()
        # TODO: how can we get system wide memory usage
        system_wide_mem_percent = psutil.memory_percent()
        print("Utilizing {} cpu percentage of a total system wide {} percent".format(cpu_percent, system_wide_cpu_percent))
        print("Utilizing {} ram percentage of a total system wide {} percent".format(mem_percent, system_wide_mem_percent))

        power_credit_cpu = cpu_percent / system_wide_cpu_percent
        power_credit_mem = mem_percent / system_wide_mem_percent

        total_power = 0
        if total_cpu_power != 0:
            total_power += total_cpu_power * power_credit_cpu
        if total_dram_power != 0:
            total_power += total_dram_power * power_credit_mem

         # assign the rest of the power to the CPU percentage even if this is a bit innacurate
        total_power += (total_intel_power - total_dram_power - total_cpu_power) * power_credit_cpu

        total_intel_power = total_intel_power * power_credit
        return total_intel_power, cpu_times, cpu_percent

def get_nvidia_gpu_power(pid_list):
    # Find per process per gpu usage info
    sp = subprocess.Popen(['nvidia-smi', 'pmon', '-c', '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_str_split = out_str[0].decode('utf-8').split('\n')
    out_str_pruned = [out_str_split[0],] + out_str_split[2:]
    out_str_final = "\n".join(out_str_pruned)
    df = pd.read_csv(pd.compat.StringIO(out_str_final[1:]), engine='python', delim_whitespace=True)
    process_percentage_used_gpu = df.groupby(['gpu','pid']).mean()

    p = Popen(['nvidia-smi', '-q', '-x'], stdout=PIPE)
    outs, errors = p.communicate()
    xml = fromstring(outs)
    num_gpus = int(xml.getiterator('attached_gpus')[0].text)
    results = []
    power = 0
    per_gpu_absolute_percent_usage = {}

    for gpu_id, gpu in enumerate(xml.getiterator('gpu')):
        gpu_data = {}

        name = gpu.getiterator('product_name')[0].text
        gpu_data['name'] = name

        # get memory
        memory_usage = gpu.getiterator('fb_memory_usage')[0]
        total_memory = memory_usage.getiterator('total')[0].text
        used_memory = memory_usage.getiterator('used')[0].text
        free_memory = memory_usage.getiterator('free')[0].text
        gpu_data['memory'] = {
            'total': total_memory,
            'used_memory': used_memory,
            'free_memory': free_memory
        }

        # get utilization
        utilization = gpu.getiterator('utilization')[0]
        gpu_util = utilization.getiterator('gpu_util')[0].text
        memory_util = utilization.getiterator('memory_util')[0].text
        gpu_data['utilization'] = {
            'gpu_util': gpu_util,
            'memory_util': memory_util
        }

        # get power
        power_readings = gpu.getiterator('power_readings')[0]
        power_draw = power_readings.getiterator('power_draw')[0].text

        gpu_data['power_readings'] = {
            'power_draw' : power_draw
        }

        # processes
        processes = gpu.getiterator('processes')[0]
        infos = []
        per_gpu_absolute_percent_usage[gpu_id] = 0 
        for info in processes.getiterator('process_info'):
            pid = info.getiterator('pid')[0].text
            process_name = info.getiterator('process_name')[0].text
            used_memory = info.getiterator('used_memory')[0].text
            gpu_based_processes = process_percentage_used_gpu[process_percentage_used_gpu['gpu'] == gpu_id]
            sm_absolute_percent = process_percentage_used_gpu[process_percentage_used_gpu['pid'] == pid]['sm']
            sm_relative_percent = float(sm_absolute_percent) / float(process_percentage_used_gpu['sm'].sum())
            per_gpu_absolute_percent_usage[gpu_id] += sm_absolute_percent
            infos.append({
                'pid': pid,
                'process_name' : process_name,
                'used_memory' : used_memory,
                'sm_relative_percent' :sm_relative_percent,
                'sm_absolute_percent' : sm_absolute_percent
            })
            if int(pid) in pid_list:
                power += sm_relative_percent * float(power_draw)
        gpu_data['processes'] = infos

        results.append(gpu_data)

    return power, per_gpu_absolute_percent_usage

# def _calculate_carbon_impact():
    # TODO: carbon impact formula here based on estimated power attribution for gpu and cpu
    # kWh = sum (Watts rapl (cpu + dram) + Watts(gpu)) * sampling interval
    # multiply by region carbon intensity

    # Probably do this for plotting not, live???? idk


def read_latest_stats(log_dir):
    log_path = os.path.join(log_dir, DATAPATH)
    last_line = subprocess.check_output(["tail", "-1", log_path])
    return last_line.split(",")


def _sample_and_log_power(log_dir):
    current_process =  psutil.Process(os.getppid())
    process_ids = [current_process.pid] + [child.pid for child in current_process.children(recursive=True) ]

    # First, try querying Intel's RAPL interface for dram at least
    rapl_power_draw, cpu_time, average_cpu_utilization = get_rapl_power(process_ids)
    nvidia_gpu_power_draw, per_gpu_absolute_percent_usage = get_nvidia_gpu_power(process_ids)
    average_gpu_utilization = np.mean(per_gpu_absolute_percent_usage.values())
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    log_path = safe_file_path(os.path.join(log_dir, DATAPATH))
    data = [timestamp, rapl_power_draw, nvidia_gpu_power_draw, cpu_time, average_gpu_utilization, average_cpu_utilization]
    write_csv_data_to_file(log_path, data)

@processify
def launch_power_monitor(log_dir):
    print("Starting process to monitor power")
    #TODO: log one time info: CPU/GPU info, version of this package, region, datetime for start of experiment, CO2 estimate data.
    # this will be used to build a latex table later.
    region, zone_info =  get_zone_information_by_coords()
    info_path = safe_file_path(os.path.join(log_dir, INFOPATH))

    data = {
        "cpu_info" : get_my_cpu_info(),
        "gpu_info" : get_gpu_info(),
        "climate_impact_tracker_version" : climate_impact_tracker.__version__,
        "region" : region,
        "experiment_start" : datetime.now(),
        "region_carbon_intensity_estimate" : zone_info # kgCO2/kWh
    }

    with open(info_path, 'wb') as info_file:
        pickle.dump(data, info_file)

    while True:
        time.sleep(SLEEP_TIME)
        _sample_and_log_power(log_dir)

