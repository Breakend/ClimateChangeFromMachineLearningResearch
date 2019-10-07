from experiment_impact_tracker.cpu.intel import get_rapl_power
from experiment_impact_tracker.cpu.common import get_hz_actual
from experiment_impact_tracker.cpu import rapl

from experiment_impact_tracker.gpu.nvidia import get_nvidia_gpu_power
from experiment_impact_tracker.utils import *

DATA_HEADERS = [
    {
        "name": "timestamp",
        "description": "Time at which sample was drawn based on local machine time in timestamp format.",
        "compatability": ["all"],
        "routing": {
            "function": get_timestamp
        }
    },
    {
        "name": "rapl_power_draw_absolute",
        "description": "The absolute power draw reading read from an Intel RAPL package. This is in terms of Watts across the entire machine.",
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power
        }
    },

    {
        "name": "rapl_estimated_attributable_power_draw",
        "description": "This is the estimated attributable power draw to this process and all child processes based on power draw reading read from an Intel RAPL package. This is calculated as (watts used by cpu) * (relative cpu percentage used) + (watts used by dram) * (relative dram percentage used) + (watts used by other package elements) * (relative cpu percentage used).",
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power
        }
    },
    {
        "name": "nvidia_draw_absolute",
        "description": "This is the absolute power draw of all accessible NVIDIA GPUs on the system (as long as the main process or any child process lives on the GPU). Calculated as sum across all GPUs.",
        "compatability": ["nvidia"],
        "routing": {
            "function": get_nvidia_gpu_power
        }
    },
    {
        "name": "nvidia_estimated_attributable_power_draw",
        "description": "This is the estimated attributable power draw of all accessible NVIDIA GPUs on the system (as long as the main process or any child process lives on the GPU). Calculated as the sum per gpu of (absolute power draw per gpu) * (relative process percent utilization of gpu)",
        "compatability": ["nvidia"],
        "routing": {
            "function": get_nvidia_gpu_power
        }
    },
    {
        "name": "cpu_time_seconds",
        "description": "This is the total CPU time used so far by the program in seconds.",
        # TODO: shouldn't need rapl, this should be available to all
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power
        }
    },
    {
        "name": "average_gpu_estimated_utilization_absolute",
        "description": "This is the absolute utilization of the GPUs by the main process and all child processes. Returns an average result across several trials of nvidia-smi pmon -c 10. Averaged across GPUs. Using .05 to indicate 5%.",
        "compatability": ["nvidia"],
        "routing": {
            "function": get_nvidia_gpu_power
        }
    },
    {
        "name": "average_gpu_estimated_utilization_relative",
        "description": "This is the relative utilization of the GPUs by the main process and all child processes. Returns an average result across several trials of nvidia-smi pmon -c 10 and the percentage that this process and all child process utilize for the gpu.  Averaged across GPUs. Using .05 to indicate 5%. ",
        "compatability": ["nvidia"],
        "routing": {
            "function": get_nvidia_gpu_power
        }
    },
    {
        "name": "average_relative_cpu_utilization",
        "description": "This is the relative CPU utlization compared to the utilization of the whole system at that time. E.g., if the total system is using 50\% of the CPU power, but our program is only using 25\%, this will return .5.",
        # TODO: shouldn't need rapl, this should be available to all
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power
        }
    },
    {
        "name": "absolute_cpu_utilization",
        "description": "This is the relative CPU utlization compared to the utilization of the whole system at that time. E.g., if the total system is using 50\% of 4 CPUs, but our program is only using 25\% of 2 CPUs, this will return .5 (same as in top). There is no multiplier times the number of cores in this case as top does. ",
        # TODO: shouldn't need rapl, this should be available to all
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power
        }
    },
    {
        "name": "per_gpu_performance_state",
        "description": "A concatenated string which gives the performance state of every single GPU used by the main process or all child processes. Example formatting looks like <gpuid>::<performance state>. E.g., 0::P0",
        "compatability": ["nvidia"],
        "routing": {
            "function": get_nvidia_gpu_power
        }
    },
    {
        "name": "relative_mem_usage",
        "description": "The percentage of all in-use ram this program is using.",
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power 
        }
    },
    {
        "name": "absolute_mem_usage",
        "description": "The amount of memory being used.",
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power
        }
    },
    {
        "name": "absolute_mem_percent_usage",
        "description": "The amount of memory being used as an absolute percentage of total memory (RAM).",
        "compatability": ["rapl"],
        "routing": {
            "function": get_rapl_power
        }
    },
    {
        "name": "hz_actual",
        "description": "The current hz of the CPU.",
        "compatability": ["all"],
        "routing": {
            "function": get_hz_actual
        }
    }
]
