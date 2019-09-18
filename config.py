
configDict = {}
i2m = 2.54/100

# Taken from SOLIDWorks "AsBuild" models in the repo.
configDict["BA"] = {
    "nbaffles" : 4, # number of forebaffles
    "gsrad" : 600/2*i2m, # in meters
    "gsheight" : 200*i2m, # in meters
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m+0.07, # in meters
    "winrad" : 0.55, # in meters
    "aptoffr" : 1.88/2, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 36.5*i2m, # in meters
    "eloffz" : 2.3, # in meters
}

# Update parameters!
configDict["Keck"] = {
    "nbaffles" : 5, # number of forebaffles
    "gsrad" : 600/2*i2m, # in meters
    "gsheight" : 200*i2m, # in meters
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m+0.07, # in meters
    "winrad" : 0.55, # in meters
    "aptoffr" : 1.88/2, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 36.5*i2m, # in meters
    "eloffz" : 2.3, # in meters
}

# Update parameters!
configDict["B3"] = {
    "nbaffles" : 1, # number of forebaffles
    "gsrad" : 600/2*i2m, # in meters
    "gsheight" : 200*i2m, # in meters
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m+0.07, # in meters
    "winrad" : 0.55, # in meters
    "aptoffr" : 0., # in meters
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

configDict["Custom"] = {
    "nbaffles" : 1, # number of forebaffles
    "gsrad" : 1e10, # in meters
    "gsheight" : 1e-1, # in meters
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m+0.07, # in meters
    "winrad" : 0.55, # in meters
    "aptoffr" : 0, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 36.5*i2m, # in meters
    "eloffz" : 2.3, # in meters
}
