import ujson as json
import multiprocessing
from shapely.geometry import shape
import os 
import numpy as np
import sys
from tqdm import tqdm

# def _read_terrible_json_lines(lines):
#    lines_fixed = []
#    for x in lines:
#        x = x.replace("/", "\/")
#        x = json.loads(x)
#        x["geometry"] = shape(x["geometry"])
#        lines_fixed.append(x)
#    return lines_fixed

#def read_terrible_json(path):
#    print("Reading lines...")
#    numlines = 10
#     
#    with open(path, 'rt') as f:
#        lines = f.readlines()

#   print("Processing {} lines...".format(len(lines)))
# reduce the result dicts into a single dict
#    tasks =  [(lines[x:min(x+numlines, len(lines))]) for x in range(0, len(lines), numlines)]

#    results = list(tqdm(map(_read_terrible_json_lines, tasks ), total=int(len(lines)/numlines)))
#    result = [item for sublist in results for item in sublist]

#    pool.close()
#    pool.join()
#    pbar.close()

#    return result

from tqdm import tqdm

def read_terrible_json(path):
    with open(path, 'rt') as f:
        lines = []
        test_read_lines = [x for x in f.readlines()]
        for x in tqdm(test_read_lines): 
            if x:
                x = x.replace("/", "\/")
                x = json.loads(x) 
                lines.append(x)
    return lines

def _load_zone_info():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'data/co2eq_parameters.json'), 'rt') as f:
        x = json.load(f)
    return x

def load_regions_with_bounding_boxes():
    print("loading region bounding boxes for computing carbon emissions region, this may take a moment...")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    all_geoms = []
    # with open('data/zone_geometries.json') as f:
    all_geoms = read_terrible_json(os.path.join(dir_path, 'data/zonegeometries.json'))
    # terribles_jsons = [
    #     'data/tmp_countries.json',
    #     'data/tmp_states.json',
    #     'data/tmp_thirdparty.json'
    #     ]
    # for terribles_json in terribles_jsons:
    #     all_geoms.extend(read_terrible_json(os.path.join(dir_path, terribles_json)))
    # for country_geo in [
    #     'data/DK-DK2-without-BHM.json',
    #     'data/NO-NO1.json',
    #     'data/NO-NO2.json',
    #     'data/NO-NO3.json',
    #     'data/NO-NO4.json',
    #     'data/NO-NO5.json',
    #     'data/SE-SE1.json',
    #     'data/SE-SE2.json',
    #     'data/SE-SE3.json',
    #     'data/SE-SE4.json',
    #     'data/sct-no-islands.json',
    #     "data/JP-CB.geojson",
    #     "data/JP-HR.geojson",
    #     "data/JP-KN.geojson",
    #     "data/JP-KY.geojson",
    #     "data/JP-ON.geojson",
    #     "data/JP-TK.geojson",
    #     "data/ES-IB-FO.geojson",
    #     "data/ES-IB-IZ.geojson",
    #     "data/ES-IB-MA.geojson",
    #     "data/ES-IB-ME.geojson",
    #     "data/AUS-TAS.geojson",
    #     "data/AUS-TAS-KI.geojson"
    # ]:
    #     with open(os.path.join(dir_path, country_geo), 'rt') as f:
    #         geo = json.load(f)
    #         all_geoms.append(geo)

    print("Turning json to shapes")
    for i, geom in enumerate(all_geoms):
        all_geoms[i]["geometry"] = shape(geom["geometry"])
    print("Done!")
    return all_geoms

REGIONS_WITH_BOUNDING_BOXES = load_regions_with_bounding_boxes()
ZONE_INFO = _load_zone_info()["fallbackZoneMixes"]
