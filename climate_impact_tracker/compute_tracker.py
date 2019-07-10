# def _get_nvidia_power_measurement():
#     measurementPower = os.popen("nvidia-smi -i 0 -q").read()
#     tmp = os.popen("ps -Af").read()
#     return nvidiaSmiParser(measurementPower, ["Power Draw"],num)

import psutil,os

class PowerTracker(object):

    def __init__(self, tensorboard_log_dir, track_power=True, track_carbon_estimate=False):
        self.track_power = track_power
        self.track_carbon_estimate = track_carbon_estimate
        self.tensorboard_log_dir = tensorboard_log_dir
    
    def _get_cpu_util(self):
        p = psutil.Process(os.getpid())
        # probably turn this into percentage and then multiply power draw
        # TODO: this probably needs to be cpu_percent of the process versus system wide percent
        # and then multiply that by power draw
        times = p.cpu_times()
        return times["user"] + times["system"]

    def _get_rapl_measurements(self):
        # supplement missing measurements with lookups 

    def _get_nvidia_measurements(self):

use this and https://github.com/wookayin/gpustat/blob/master/gpustat/core.py to track process usage over time, combined with readouts for power from this and intel thingy