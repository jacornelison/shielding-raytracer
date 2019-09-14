
configDict = {}

configDict["BA"] = {
    "gsrad" : 600/2*2.54/100, # in meters
    "gsheight" : 200*2.54/100, # in meters
    "fbrad": 48/2*2.54/100, # in meters
    "fbheight": 40*2.54/100, # in meters
    "aptoffr" : 30*2.54/100, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 1, # in meters
    "eloffz" : 1, # in meters
}

# Get the forebaffles and gs out of the way so we can look at a sphere
configDict["test"] = {
    "gsrad" : 1e10, # in meters
    "gsheight" : 0.0001, # in meters
    "fbrad": 1e4-1, # in meters
    "fbheight": 1, # in meters
    "aptoffr" : 0, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 0, # in meters
    "eloffz" : 0, # in meters
}

