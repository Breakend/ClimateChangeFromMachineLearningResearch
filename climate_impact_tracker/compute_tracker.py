# def _get_nvidia_power_measurement():
#     measurementPower = os.popen("nvidia-smi -i 0 -q").read()
#     tmp = os.popen("ps -Af").read()
#     return nvidiaSmiParser(measurementPower, ["Power Draw"],num)

import psutil,os
import .rapl
import time
import subprocess
import pandas as pd
import threading

from datetime import datetime

from subprocess import Popen, PIPE
from xml.etree.ElementTree import fromstring

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
        for process in pid_list:
            p = psutil.Process(os.getpid())
            cpu_percent += p.cpu_percent()
            mem_percent += p.memory_percent()

        system_wide_cpu_percent = psutil.cpu_percent()
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
        return total_intel_power

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
        for info in processes.getiterator('process_info'):
            pid = info.getiterator('pid')[0].text
            process_name = info.getiterator('process_name')[0].text
            used_memory = info.getiterator('used_memory')[0].text
            gpu_based_processes = process_percentage_used_gpu[process_percentage_used_gpu['gpu'] == gpu_id]
            sm_absolute_percent = process_percentage_used_gpu[process_percentage_used_gpu['pid'] == pid]['sm']
            sm_relative_percent = float(sm_absolute_percent) / float(process_percentage_used_gpu['sm'].sum())
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

    return power




class PowerTracker(object):

    def __init__(self, tensorboard_log_dir, track_power=True, track_carbon_estimate=False):
        # TODO: per process power or per machine power flag, per machine is likely more accurate 
        # if the only job on the machine (not including the slurm cluster which counts all cpu usage)
        # for the node in RAPL
        self.track_power = track_power
        self.track_carbon_estimate = track_carbon_estimate
        self.tensorboard_log_dir = tensorboard_log_dir
        current_process = psutil.Process()
        self.all_processes = [current_process.pid] + [child.pid for child in current_process.children(recursive=True) ]

    def _sample_power(self):
        # First, try querying Intel's RAPL interface for dram at least
        rapl_power_draw = get_rapl_power(self.all_processes)
        nvidia_gpu_power_draw = get_nvidia_gpu_power(self.all_processes)
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        log_path = os.path.join(self.tensorboard_log_dir, 'power_measurements/power.csv')

        with open(log_path, 'a+') as f:
            if os.path.exists(log_path):
                f.write("timestamp, rapl_power_sample,nvidia_gpu_power_sample\n")
            f.write(timestamp + "," + str(rapl_power_draw) + "," + str(nvidia_gpu_power_draw) + "\n")

    def _continuously_monitor(self):
        while True:
            time.sleep(10)
            self._sample_power()
    
    def run(self):
        thread = threading.Thread(target=self._continuously_monitor)
        thread.start()
        print("Spun off power monitoring thread.")