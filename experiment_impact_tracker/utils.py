import atexit
import csv
import os
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd

import psutil

_timer = getattr(time, 'monotonic', time.time)
def get_timestamp(*args, **kwargs):
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    return timestamp

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
