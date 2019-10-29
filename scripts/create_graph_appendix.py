import os.path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import random
import string

"""
Usage

input_path will be where the data.csv file lives

create_graphs(input_path: str, csv='data.csv', output_path: str ='.', fig_x:int =16, fig_y:int = 8)
"""


def random_suffix(length=4):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def dateparse (time_in_secs):    
    return datetime.datetime.fromtimestamp(float(time_in_secs))

def clean(x): #2.9066 GHz
    x = x.replace(" GHz", "")
    return float(x)

HZ_ACTUAL_COL = 'hz_actual'
TIMESTAMP_COL = 'timestamp'

SKIP_COLUMN = ['per_gpu_performance_state']

# TODO move per_gpu_performance_state to special handler
SPECIAL_COLUMN = [HZ_ACTUAL_COL]

HANDLER_MAP = {HZ_ACTUAL_COL: clean}


def create_graphs(input_path: str, csv='data.csv', output_path: str ='.', fig_x:int =16, fig_y:int = 8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # create graph dirs
    graph_dir = csv + '_' + str(fig_x) + '_' + str(fig_y)
    out_dir = os.path.join(output_path, graph_dir)
    if os.path.exists(out_dir):
        print("{} already exists, attaching random string to the out put dir and moving on.".format(out_dir))
        out_dir = out_dir + '_' + random_suffix()

    os.makedirs(out_dir)
    
    df = pd.read_csv(os.path.join(input_path, csv), sep=',', parse_dates=[0], date_parser=dateparse)
    for k in list(df)[1:]:
        if k in SKIP_COLUMN:
            continue
        if k in SPECIAL_COLUMN:
            df[k] = df[k].apply(HANDLER_MAP[k])
        df.plot(kind='line', x=TIMESTAMP_COL, y=k, figsize=(25,8))
        plt.savefig(os.path.join(out_dir, k+'.png'))