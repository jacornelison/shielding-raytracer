
configDict = {}
i2m = 2.54/100
configDict["BA"] = {
    "nbaffles" : 4, # number of forebaffles
    "gsrad" : 600/2*i2m, # in meters
    "gsheight" : 200*i2m, # in meters
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m, # in meters
    "aptoffr" : 1.88/2, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 36.5*i2m, # in meters
    "eloffz" : 2.3, # in meters
}

# Get the forebaffles and gs out of the way so we can look at a sphere
configDict["test"] = {
    "nbaffles" : 0, # number of forebaffles
    "gsrad" : 1e10, # in meters
    "gsheight" : 0.0001, # in meters
    "fbrad": 1e4-1, # in meters
    "fbheight": 1, # in meters
    "aptoffr" : 0, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 0, # in meters
    "eloffz" : 0, # in meters
}

