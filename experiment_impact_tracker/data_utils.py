import pickle
import ujson as json
from pandas.io.json import json_normalize
from datetime import datetime
import os
import csv

BASE_LOG_PATH = 'impacttracker/'
DATAPATH = BASE_LOG_PATH + 'data.json'
INFOPATH = BASE_LOG_PATH + 'info.pkl'

def load_initial_info(log_dir):
    info_path = safe_file_path(os.path.join(log_dir, INFOPATH))
    with open(info_path, 'rb') as info_file:
        return pickle.load(info_file)

def _read_json_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        return [json.loads(line) for line in lines]

def load_data_into_frame(log_dir):
    data_path = safe_file_path(os.path.join(log_dir, DATAPATH))
    json_array = _read_json_file(data_path)
    return json_normalize(json_array), json_array

def log_final_info(log_dir):
    final_time = datetime.now()
    info = load_initial_info(log_dir)
    info["experiment_end"] = final_time
    info_path = safe_file_path(os.path.join(log_dir, INFOPATH))

    with open(info_path, 'wb') as info_file:
        pickle.dump(info, info_file)

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

def write_json_data_to_file(file_path, data, overwrite=False):
    file_path = safe_file_path(file_path)
    with open(file_path, 'w' if overwrite else 'a') as outfile:
        outfile.write(ujson.dumps(data) + "\n")
