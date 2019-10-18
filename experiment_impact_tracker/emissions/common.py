import experiment_impact_tracker.emissions.us_ca_parser as us_ca_parser

REALTIME_REGIONS = {
    "US-CA" : us_ca_parser
}

def is_capable_realtime_carbon_intensity(region):
    return region in REALTIME_REGIONS
    
def get_realtime_carbon_source(region):
    return REALTIME_REGIONS[region].get_realtime_carbon_source()

def get_realtime_carbon(*args, **kwargs):
    if 'region' not in kwargs:
        raise ValueError("region was not passed to function")
    carbon_intensity = REALTIME_REGIONS[kwargs['region']].fetch_supply()[0]['carbon_intensity']
    print(carbon_intensity)
    return {
        "realtime_carbon_intensity" : carbon_intensity 
    }
