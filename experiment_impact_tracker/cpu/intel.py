import atexit
import os
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import cpuinfo
import psutil
from experiment_impact_tracker.cpu.common import get_my_cpu_info
from experiment_impact_tracker.utils import *

from . import rapl


def get_and_cache_cpu_max_tdp_from_intel():
    # Realized this isn't really worth anything because TDP isn't a reliable estimator of power output
    cpu_brand = cpuinfo.get_cpu_info()['brand'].split(' ')[2]
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuinfocache/{}'.format(cpu_brand))):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuinfocache/{}'.format(cpu_brand)), 'r') as f:
            return int(f.readline())
    s = requests.Session()
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36'
    s.headers['User-Agent'] = user_agent
    r = s.get('https://ark.intel.com/content/www/us/en/ark/search.html?_charset_=UTF-8&q={}'.format(
        cpu_brand), allow_redirects=True)
    soup = BeautifulSoup(r.content, 'lxml')
    results = soup.find_all('span', attrs={'data-key': "MaxTDP"})

    if len(results) == 0:
        redirect_url = soup.find(id='FormRedirectUrl').attrs['value']
        if redirect_url:
            r = s.get("https://ark.intel.com/" +
                      redirect_url, allow_redirects=True)
            soup = BeautifulSoup(r.content, 'lxml')
            results = soup.find_all('span', attrs={'data-key': "MaxTDP"})

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuinfocache/{}'.format(cpu_brand)), 'w') as f:
        f.write((results[0].text.strip().replace('W', '')))
    return int(results[0].text.strip().replace('W', ''))


_timer = getattr(time, 'monotonic', time.time)


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
            raise ValueError(
                "Unexpected top level domain for RAPL package. Not yet supported.")

        total_intel_power += power

        for sd in domain.subdomains:
            subdomain = domain.subdomains[sd]
            power = diff.average_power(
                package=domain.name, domain=subdomain.name)
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
        raise ValueError(
            "Don't support credit assignment to Intel RAPL GPU yet.")

    cpu_percent = 0
    absolute_cpu_percent = 0
    cpu_times = 0
    mem_infos = []
    for process in pid_list:
        try:
            p = psutil.Process(process)
        except psutil.NoSuchProcess:
            if logger is not None:
                logger.warn(
                    "Process with pid {} used to be part of this process chain, but was shut down. Skipping.")
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
        delta_proc2 = (system_wide_pt2.user - system_wide_pt1.user) + \
            (system_wide_pt2.system - system_wide_pt1.system)

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
    system_wide_mem_percent = (
        np.sum([x.rss for x in mem_infos]) / float(total_physical_memory.used))

    power_credit_cpu = cpu_percent  # / system_wide_cpu_percent
    power_credit_mem = system_wide_mem_percent
    if power_credit_cpu == 0:
        raise ValueError(
            "Problem retrieving CPU usage percentage to assign power credit")
    if power_credit_mem == 0:
        raise ValueError(
            "Problem retrieving Mem usage percentage to assign power credit")

    total_attributable_power = 0
    if total_cpu_power != 0:
        total_attributable_power += total_cpu_power * power_credit_cpu
    if total_dram_power != 0:
        total_attributable_power += total_dram_power * power_credit_mem

        # assign the rest of the power to the CPU percentage even if this is a bit innacurate
    total_attributable_power += (total_intel_power -
                                 total_dram_power - total_cpu_power) * power_credit_cpu

    if total_intel_power == 0:
        raise ValueError(
            "It seems that power estimates from Intel RAPL are coming back 0, this indicates a problem.")

    data_return_values_with_headers = {
        "rapl_power_draw_absolute": total_intel_power,
        "rapl_estimated_attributable_power_draw": total_attributable_power,
        "cpu_time_seconds": cpu_times,
        "average_relative_cpu_utilization": cpu_percent,
        "absolute_cpu_utilization": absolute_cpu_percent
    }

    return data_return_values_with_headers
