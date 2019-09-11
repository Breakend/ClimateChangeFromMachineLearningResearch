import sys
import argparse
from experiment_impact_tracker.get_region_metrics import get_zone_information_by_coords
import json
def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=
        """
        This is a test of the command line argument parser in Python.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("cloud_provider")
    return(p.parse_args())


    
# Try running with these args
#
# "Hello" 123 --enable
if __name__ == '__main__':
    
    if sys.version_info<(3,0,0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)
        
    args = cmdline_args()

    if args.cloud_provider == "azure":
        with open('./experiment_impact_tracker/data/azure_regions.json', 'rb') as f:
            azure_regions = json.load(f)
        
        for region in azure_regions:
            print(region["name"])
            information = get_zone_information_by_coords((float(region['latitude']), float(region['longitude'])))
            print(information)
    elif args.cloud_provider == "aws":
        with open('./experiment_impact_tracker/data/aws_regions.csv', 'rt') as f:
            aws_regions = f.readlines()
        
        for region in aws_regions:
            name, city, lat, lon = region.strip().split(";")
            print(name)
            information = get_zone_information_by_coords((float(lat), float(lon)))
            print(information)       
    elif args.cloud_provider == "gcp":
        with open('./experiment_impact_tracker/data/gcp_regions.csv', 'rt') as f:
            gcp_regions = f.readlines()
        
        for region in gcp_regions:
            name, city, lat, lon = region.strip().split(";")
            print(name)
            information = get_zone_information_by_coords((float(lat), float(lon)))
            print(information)      
    else:
        raise ValueError("region not supported") 
