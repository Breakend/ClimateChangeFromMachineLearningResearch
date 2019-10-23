import sys
import argparse
from experiment_impact_tracker.get_region_metrics import get_current_region_info, get_sorted_region_infos, get_zone_information_by_coords
from pprint import pprint
from geopy.geocoders import Nominatim

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=
        """
        This is a test of the command line argument parser in Python.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("command", help="The thing to run from: [current, top, bottom, longlat, address]")
    p.add_argument("--n", type=int)
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--address", type=str)
    # p.add_argument("required_int", type=int,
    #                help="req number")
    # p.add_argument("--on", action="store_true",
    #                help="include to enable")
    # p.add_argument("-v", "--verbosity", type=int, choices=[0,1,2], default=0,
    #                help="increase output verbosity")
                   
    # group1 = p.add_mutually_exclusive_group(required=True)
    # group1.add_argument('--enable',action="store_true")
    # group1.add_argument('--disable',action="store_false")

    return(p.parse_args())


    
# Try running with these args
#
# "Hello" 123 --enable
if __name__ == '__main__':
    
    if sys.version_info<(3,0,0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)
        
    args = cmdline_args()
    if args.command == "current":
        pprint(get_current_region_info())
    elif args.command == "top":
        pprint(get_sorted_region_infos()[:args.n])
    elif args.command == "bottom":
        pprint(get_sorted_region_infos()[-args.n:])
    elif args.command == "longlat":
        pprint(get_zone_information_by_coords((args.lat, args.lon)))
    elif args.command == "address":
        geolocator = Nominatim(user_agent="experiment_impact_tracker")
        location = geolocator.geocode(args.address)
        pprint(get_zone_information_by_coords((location.latitude, location.longitude)))