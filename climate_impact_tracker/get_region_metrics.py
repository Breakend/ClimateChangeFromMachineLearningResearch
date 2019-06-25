from .constants import REGIONS_WITH_BOUNDING_BOXES, ZONE_INFO
from shapely.geometry import Point

def get_zone_information_by_coords(coords):
    region = get_region_by_coords(coords)
    return region, ZONE_INFO[region["id"]]

def get_region_by_coords(coords):
    #TODO: automatically narrow down possibilities
    lat, lon = coords
    point = Point(lon, lat)
    zone_possibilities = []
    for zone in REGIONS_WITH_BOUNDING_BOXES:
        try:
            if zone["geometry"].contains(point):
                zone_possibilities.append(zone)
        except:
            import pdb; pdb.set_trace()    
    if len(zone_possibilities) == 0:
        raise ValueError("No possibilities found, may need to add a zone.")
        
    z = min(zone_possibilities, key=lambda x: x["geometry"].area)
    return z

def get_current_location():
    import geocoder
    g = geocoder.ip('me')
    return g.y, g.x

def get_current_region_info():
    return get_zone_information_by_coords(get_current_location())