import experiment_impact_tracker.emissions.us_ca_parser as us_ca_parser

REALTIME_REGIONS = {
    "US-CA" : us_ca_parser.fetch_supply
}

def is_capable_realtime_carbon_intensity(region):
    return region in REALTIME_REGIONS:
    

def get_realtime_carbon(*args, **kwargs):
    if 'region' not in kwargs:
        raise ValueError("region was not passed to function")

    return {
        "realtime_carbon_intensity" : REALTIME_REGIONS[kwargs['region']][0]
    }
