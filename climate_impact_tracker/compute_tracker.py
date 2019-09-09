import psutil,os
import atexit
import pickle
import numpy as np
from climate_impact_tracker import rapl
import time
import subprocess
import pandas as pd
import threading
import time
import logging
_timer = getattr(time, 'monotonic', time.time)

from datetime import datetime

from subprocess import Popen, PIPE
from xml.etree.ElementTree import fromstring

import os
import sys
import traceback
from functools import wraps
from .processor_info import get_my_cpu_info, get_gpu_info
from multiprocessing import Process, Queue
from queue import Empty as EmptyQueueException
import csv
from collections import OrderedDict



BASE_LOG_PATH = 'impacttracker/'
DATAPATH = BASE_LOG_PATH + 'data.csv'
INFOPATH = BASE_LOG_PATH + 'info.pkl'
DATA_HEADERS = ["timestamp","rapl_power_draw_absolute", "rapl_estimated_attributable_power_draw", "nvidia_draw_absolute", "nvidia_estimated_attributable_power_draw", "cpu_time_seconds", "average_gpu_estimated_utilization_absolute", "average_gpu_estimated_utilization_relative", "average_relative_cpu_utilization", "absolute_cpu_utilization", "per_gpu_performance_state"]
SLEEP_TIME = 1
PUE = 1.58 

def get_flop_count_tensorflow(graph=None, freeze_graph=False, name_spaces=[]):
    import tensorflow as tf # import within function so as not to require tf for package
    from tensorflow.python.framework import graph_util
    if graph is None:
        graph = tf.get_default_graph()

    if freeze_graph:
        with tf.Session() as sess:
            output_graph_def = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), name_spaces)
            with tf.gfile.GFile('/tmp/tmp_flop_count_graph.pb', "wb") as f:
                f.write(output_graph_def.SerializeToString())
            g2 = load_pb('./graph.pb')
            with g2.as_default():
                flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())

    else:
        flops = tf.profiler.profile(graph, options = tf.profiler.ProfileOptionBuilder.float_operation())
    return flops.total_float_ops

def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(q, *args, **kwargs)
        except Exception as e:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
            q.put((ret, error))
            raise e
        else:
            error = None
        q.put((ret, error))


    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        queue = Queue() # not the same as a Queue.Queue()
        p = Process(target=process_func, args=[queue] + list(args), kwargs=kwargs)
        p.start()
        return p, queue
    return wrapper


def safe_file_path(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path

def write_csv_data_to_file(file_path, data, overwrite=False):
    file_path = safe_file_path(file_path)
    with open(file_path, 'w' if overwrite else 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

def get_rapl_power(pid_list, logger=None):
        s1 = rapl.RAPLMonitor.sample()
        time.sleep(3)
        s2 = rapl.RAPLMonitor.sample()
        diff = s2 - s1
        total_intel_power = 0
        total_dram_power = 0
        total_cpu_power = 0
        total_gpu_power = 0
        for d in diff.domains:
            domain = diff.domains[d]
            power = diff.average_power(package=domain.name)
            # this should get the power per package (e.g., total rapl power)
            # see images/power-planes.png for example
            # Downloaded from: https://blog.chih.me/images/power-planes.jpg
            #  Recent (Sandy Bridge and later) Intel processors that implement the RAPL (Running Average Power Limit) 
            # interface that provides MSRs containing energy consumption estimates for up to four power planes or 
            # domains of a machine, as seen in the diagram above.
            # PKG: The entire package.
            # PP0: The cores.
            # PP1: An uncore device, usually the GPU (not available on all processor models.)
            # DRAM: main memory (not available on all processor models.)
            # The following relationship holds: PP0 + PP1 <= PKG. DRAM is independent of the other three domains.
            # Most processors come in two packages so top level domains shold be package-1 and package-0
            if "package" not in domain.name:
                raise ValueError("Unexpected top level domain for RAPL package. Not yet supported.")

            total_intel_power += power

            for sd in domain.subdomains:
                subdomain = domain.subdomains[sd]
                power = diff.average_power(package=domain.name, domain=subdomain.name)
                subdomain = subdomain.name.lower()
                if subdomain == "ram" or subdomain == "dram":
                    total_dram_power += power
                elif subdomain == "cores" or subdomain == "cpu":
                    total_cpu_power += power
                elif subdomain == "gpu":
                    total_gpu_power += power
                # other domains get don't have relevant readouts to give power attribution, therefore
                # will get assigned the same amount of credit as the CPU

        if total_gpu_power != 0:
            raise ValueError("Don't support credit assignment to Intel RAPL GPU yet.")

        cpu_percent = 0
        absolute_cpu_percent = 0
        cpu_times = 0
        mem_infos = []
        for process in pid_list:
            try:
                p = psutil.Process(process)
            except psutil.NoSuchProcess:
                if logger is not None:
                    logger.warn("Process with pid {} used to be part of this process chain, but was shut down. Skipping.")
                continue
            # Modifying code https://github.com/giampaolo/psutil/blob/c10df5aa04e1ced58d19501fa42f08c1b909b83d/psutil/__init__.py#L1102-L1107 
            # We want relative percentage of CPU used so we ignore the multiplier by number of CPUs, we want a number from 0-1.0 to give
            # power credits accordingly
            st1 = _timer()
            # units in terms of cpu-time, so we need the cpu in the last time period that are for the process only
            system_wide_pt1 = psutil.cpu_times()
            pt1 = p.cpu_times()
            time.sleep(1)
            st2 = _timer()
            system_wide_pt2 = psutil.cpu_times()
            pt2 = p.cpu_times()

            # change in cpu-hours process
            delta_proc = (pt2.user - pt1.user) + (pt2.system - pt1.system)
            # change in cpu-hours system
            delta_proc2 = (system_wide_pt2.user - system_wide_pt1.user) + (system_wide_pt2.system - system_wide_pt1.system)

            # percent of cpu-hours in time frame attributable to this process (e.g., attributable compute)
            attributable_compute = delta_proc / float(delta_proc2)
        
            delta_time = st2 - st1

            # cpu-seconds / seconds = cpu util 
            # NOTE: WE DO NOT MULTIPLY BY THE NUMBER OF CORES LIKE HTOP, WE WANT 100% to be the max
            # since we want a percentage of the total packages. 
            # TODO: I'm not sure if this will get that in all configurations of hardware.
            absolute_cpu_percent += delta_proc / float(delta_time)

            # TODO: do we really need to do anything with the time units?
            cpu_percent += attributable_compute

            # only care about cpu_times for latest number 
            cpu_times += (pt2.user) + (pt2.system)
            mem_infos.append(p.memory_full_info())

        system_wide_cpu_percent = psutil.cpu_percent(interval=1)
        # TODO: how can we get system wide memory usage
        total_physical_memory = psutil.virtual_memory()
        # what percentage of used memory can be attributed to this process
        system_wide_mem_percent = (np.sum([x.rss for x in mem_infos]) / float(total_physical_memory.used))

        power_credit_cpu = cpu_percent #/ system_wide_cpu_percent
        power_credit_mem = system_wide_mem_percent 
        if power_credit_cpu == 0:
            raise ValueError("Problem retrieving CPU usage percentage to assign power credit")
        if power_credit_mem == 0:
            raise ValueError("Problem retrieving Mem usage percentage to assign power credit")

        total_attributable_power = 0
        if total_cpu_power != 0:
            total_attributable_power += total_cpu_power * power_credit_cpu
        if total_dram_power != 0:
            total_attributable_power += total_dram_power * power_credit_mem

         # assign the rest of the power to the CPU percentage even if this is a bit innacurate
        total_attributable_power += (total_intel_power - total_dram_power - total_cpu_power) * power_credit_cpu

        if total_intel_power == 0:
            raise ValueError("It seems that power estimates from Intel RAPL are coming back 0, this indicates a problem.")

        return total_intel_power, total_attributable_power, cpu_times, cpu_percent, absolute_cpu_percent

def get_nvidia_gpu_power(pid_list, logger=None):
    # Find per process per gpu usage info
    sp = subprocess.Popen(['nvidia-smi', 'pmon', '-c', '10'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_str_split = out_str[0].decode('utf-8').split('\n')
    # sometimes with too many processess on the machine or too many gpus, this command will reprint the headers
    # to avoid that we just remove duplicate lines
    out_str_split = list(OrderedDict.fromkeys(out_str_split))
    out_str_pruned = [out_str_split[0],] + out_str_split[2:]
    out_str_final = "\n".join(out_str_pruned)
    out_str_final = out_str_final.replace("-","0")
    df = pd.read_csv(pd.compat.StringIO(out_str_final[1:]), engine='python', delim_whitespace=True)
    process_percentage_used_gpu = df.groupby(['gpu','pid']).mean().reset_index()

    p = Popen(['nvidia-smi', '-q', '-x'], stdout=PIPE)
    outs, errors = p.communicate()
    xml = fromstring(outs)
    num_gpus = int(xml.findall('attached_gpus')[0].text)
    results = []
    power = 0
    per_gpu_absolute_percent_usage = {}
    per_gpu_relative_percent_usage  = {}
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
            'power_draw' : power_draw
        }
        absolute_power += float(power_draw.replace("W", ""))

        # processes
        processes = gpu.findall('processes')[0]

        infos = []
        # all the info for processes on this particular gpu that we're on
        gpu_based_processes = process_percentage_used_gpu[process_percentage_used_gpu['gpu'] == gpu_id]
        # what's the total absolute SM for this gpu across all accessible processes 
        percentage_of_gpu_used_by_all_processes = float(gpu_based_processes['sm'].sum())
       
        for info in processes.findall('process_info'):
            pid = info.findall('pid')[0].text
            process_name = info.findall('process_name')[0].text
            used_memory = info.findall('used_memory')[0].text
            sm_absolute_percent = gpu_based_processes[gpu_based_processes['pid'] == int(pid)]['sm'].sum()
            if percentage_of_gpu_used_by_all_processes == 0:
                # avoid divide by zero, sometimes nothing is used so 0/0 should = 0 in this case
                sm_relative_percent = 0
            else:
                sm_relative_percent = sm_absolute_percent / percentage_of_gpu_used_by_all_processes 
            infos.append({
                'pid': pid,
                'process_name' : process_name,
                'used_memory' : used_memory,
                'sm_relative_percent' :sm_relative_percent,
                'sm_absolute_percent' : sm_absolute_percent
            })

            if int(pid) in pid_list:
                # only add a gpu to the list if it's being used by one of the processes. sometimes nvidia-smi seems to list all gpus available 
                # even if they're not being used by our application, this is a problem in a slurm setting
                if gpu_id not in per_gpu_absolute_percent_usage:
                    per_gpu_absolute_percent_usage[gpu_id] = 0 #percentage_of_gpu_used_by_all_processes 
                if gpu_id not in per_gpu_relative_percent_usage:
                     per_gpu_relative_percent_usage[gpu_id] = 0 #percentage_of_gpu_used_by_all_processes 

                if gpu_id not in per_gpu_performance_states:
                    # we only log information for gpus that we're using, we've noticed that nvidia-smi will sometimes return information
                    # about all gpu's on a slurm cluster even if they're not assigned to a worker 
                    performance_state = gpu.findall('performance_state')[0].text
                    per_gpu_performance_states[gpu_id] = performance_state

                power += sm_relative_percent * float(power_draw.replace("W", ""))
                per_gpu_absolute_percent_usage[gpu_id] += (sm_absolute_percent / 100.0) # want a proportion value rather than percentage
                per_gpu_relative_percent_usage[gpu_id] += sm_relative_percent

        gpu_data['processes'] = infos

        results.append(gpu_data)
        

    return absolute_power, power, per_gpu_absolute_percent_usage, per_gpu_relative_percent_usage, per_gpu_performance_states

# def _calculate_carbon_impact():
    # TODO: carbon impact formula here based on estimated power attribution for gpu and cpu
    # kWh = sum (Watts rapl (cpu + dram) + Watts(gpu)) * sampling interval
    # multiply by region carbon intensity

    # Probably do this for plotting not, live???? idk


def read_latest_stats(log_dir):
    log_path = os.path.join(log_dir, DATAPATH)

    last_line = str(subprocess.check_output(["tail", "-1", log_path]))

    if last_line:
        return last_line.split(",")
    else:
        return None

def _stringify_performance_states(state_dict):
    return "|".join("::".join(map(lambda x: str(x), z)) for z in state_dict.items())

def _sample_and_log_power(log_dir, logger=None):
    current_process =  psutil.Process(os.getppid())
    process_ids = [current_process.pid] + [child.pid for child in current_process.children(recursive=True) ]

    # First, try querying Intel's RAPL interface for dram at least
    rapl_power_draw_absolute, rapl_draw_attributable, cpu_time, average_cpu_utilization, absolute_cpu_utilization = get_rapl_power(process_ids, logger=logger)
    nvidia_power_draw_absolute, nvidia_gpu_power_draw, per_gpu_absolute_percent_usage, per_gpu_relative_percent_usage, per_gpu_performance_states = get_nvidia_gpu_power(process_ids, logger=logger)
    average_gpu_utilization = np.mean(list(per_gpu_absolute_percent_usage.values()))
    average_gpu_relative_utilization = np.mean(list(per_gpu_relative_percent_usage.values()))
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    log_path = safe_file_path(os.path.join(log_dir, DATAPATH))
    data = [timestamp, rapl_power_draw_absolute, rapl_draw_attributable, nvidia_power_draw_absolute, nvidia_gpu_power_draw, cpu_time, average_gpu_utilization, average_gpu_relative_utilization, average_cpu_utilization, absolute_cpu_utilization, _stringify_performance_states(per_gpu_performance_states)]
    write_csv_data_to_file(log_path, data)

@processify
def launch_power_monitor(queue, log_dir, logger=None):
    logger.warn("Starting process to monitor power")
    while True:
        try:
            message = queue.get(block=False)
            if isinstance(message, str):
                if message == "Stop": return
            else:
                queue.put(message)
        except EmptyQueueException:
            pass

        try:
            _sample_and_log_power(log_dir, logger)
        except:
            ex_type, ex_value, tb = sys.exc_info()
            logger.error("Encountered exception within power monitor thread!")
            logger.error(''.join(traceback.format_tb(tb)))
            raise 
        time.sleep(SLEEP_TIME)
   

def gather_initial_info(log_dir):
    #TODO: log one time info: CPU/GPU info, version of this package, region, datetime for start of experiment, CO2 estimate data.
    # this will be used to build a latex table later.
    
    from climate_impact_tracker.get_region_metrics import get_current_region_info
    import climate_impact_tracker
    region, zone_info =  get_current_region_info()
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

    # touch datafile to clear out any past cruft and write headers
    data_path = safe_file_path(os.path.join(log_dir, DATAPATH))
    write_csv_data_to_file(data_path, DATA_HEADERS, overwrite=True) 


def load_initial_info(log_dir):
    info_path = safe_file_path(os.path.join(log_dir, INFOPATH))
    with open(info_path, 'rb') as info_file:
        return pickle.load(info_file)

def load_data_into_frame(log_dir):
    data_path = safe_file_path(os.path.join(log_dir, DATAPATH))
    return pd.read_csv(data_path)
    


class ImpactTracker(object):

    def __init__(self, logdir):
        self.logdir = logdir
        self._setup_logging()
        self.logger.warn("Gathering system info for reproducibility...")
        gather_initial_info(logdir)
        self.logger.warn("Done initial setup and information gathering...")


    def _setup_logging(self):
        # Create a custom logger
        logger = logging.getLogger("climate_impact_tracker.compute_tracker.ImpactTracker")

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(safe_file_path(os.path.join(self.logdir, BASE_LOG_PATH, 'impact_tracker_log.log')))
        c_handler.setLevel(logging.WARNING)
        f_handler.setLevel(logging.ERROR)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        self.logger = logger

    def launch_impact_monitor(self):
        try:
            self.p, self.queue = launch_power_monitor(self.logdir, self.logger)
            atexit.register(lambda p: p.terminate(), self.p)
        except:
            ex_type, ex_value, tb = sys.exc_info()
            self.logger.error("Encountered exception when launching power monitor thread.")
            self.logger.error(ex_type, ex_value, ''.join(traceback.format_tb(tb)))
            raise 

    def get_latest_info_and_check_for_errors(self):
        try:
            message = self.queue.get(block=False)
            if isinstance(message, tuple):
                ret, error = message
            else:
                self.queue.put(message)
            if error:
                ex_type, ex_value, tb_str = error
                message = '%s (in subprocess)\n%s' % (str(ex_value), tb_str)
                raise ex_type(message)
        except EmptyQueueException:
            # Nothing in the message queue
            pass
        # TODO: make thread safe read/writes via multiprocessing lock. 
        # There might be a case where we try to read a file that is currently being written to? possibly
        return read_latest_stats(self.logdir)
