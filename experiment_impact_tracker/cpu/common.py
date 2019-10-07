import cpuinfo
from .exceptions import CPUAttributeAssertionError


def get_my_cpu_info():
    return cpuinfo.get_cpu_info()

def get_hz_actual(*args, **kwargs):
    return cpuinfo.get_cpu_info()['hz_actual']

def assert_cpus_by_attributes(attributes_set):
    """Assert that you're running on CPUs with a certain set of attributes.

    This helps when running jobs in a cluster setting with heterogeneous CPUs
    to filter out sets of CPUs that you'd rather avoid. Example attributes:

    { 
        "brand": "Intel(R) Xeon(R) CPU E5-2640 v3 @ 2.60GHz",
        "hz_advertised": "2.6000 GHz"
    }

    Args:
        attributes_set (dict): set of attribute key pairs

    Raises:
        CPUAttributeAssertionError on encountered asserted attribute mismatch
    """
    cpu_info = get_my_cpu_info()
    for attribute, value in attributes_set.items():
        try:
            if cpu_info[attribute] != value:
                raise CPUAttributeAssertionError("Attribute {} asserted to be {}, but found {} instead.".format(
                    attribute, value, cpu_info[attribute]))
        except KeyError:
            raise CPUAttributeAssertionError("Attribute {} does not exist. Available attributes: {}.".format(
                attribute, ",".join(list(cpu_info.keys()))))
