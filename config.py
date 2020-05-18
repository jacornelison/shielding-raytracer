
configDict = {}
i2m = 2.54/100 #inches to meters
d2r = 0.0175 #degrees to radians
# 13 Sep 2019
# Taken from SOLIDWorks "AsBuild" models in the repo.
# FB dimensions from Fig. 3.7 of NWP's 20190207_BA_Forebaffle_Concepts posting in BA logbook.
configDict["BA"] = {
    "nbaffles" : 4, # number of forebaffles
#    "gsrad" : [2.79,5.05,7.34], # in meters
    "gsrad": 0,  # in meters
    "gsheight" : [0,0.94,3.97], # in meters
    "gssides" : 8,
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m+0.07, # in meters, 0.07 is dist from FB bottom to window.
    "winrad" : 0.69/2, # in meters
    "aptoffr" : 1.88/2, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 36.5*i2m, # in meters
    "dkoffx": 0.0,  # in meters
    "dkoffy": 0.0,  # in meters
    "eloffx": 0.0,  # in meters
    "eloffz" : 2.3, # in meters
}

# 23 Sep 2019
# Mount dimensions taken from Colin's pointing model defaults. Can't find mount in repo.
# FB dimensions from Michael Gordon's Forebaffle construction write up
configDict["keck"] = {
    "nbaffles" : 5, # number of forebaffles
    #"gsrad" : 600/2*i2m, # in meters
    #"gsheight" : 200*i2m, # in meters
    "gsrad" : 0, # in meters
    #"gsrad" : [2.79,5.05,7.34], # in meters
    "gsheight" : [0,0.94,3.97], # in meters
    "gssides" : 8,
    "fbrad": 25/2*i2m, # in meters
    "fbheight": 29*i2m, # in meters
    "winrad" : 16.1/2*i2m, # in meters. #from membrane ring Dia.
    "aptoffr" : 0.5458, # in meters
    "drumangle" : 211*d2r, # RADIANS
    "aptoffz" : 1.5964, # in meters
    "dkoffx" : -1.0196, # in meters
    "dkoffy" : 0.0, # in meters
    "eloffx" : 0.0, # in meters
    "eloffz" : 1.1750, # in meters
}


# Update parameters!
configDict["B3"] = {
    "nbaffles" : 1, # number of forebaffles
    "gsrad" : 600/2*i2m, # in meters
    "gsheight" : 200*i2m, # in meters
    "gssides" : 8,
    "fbrad": 48/2*i2m, # in meters
    "fbheight": 40*i2m+0.07, # in meters
    "winrad" : 0.69/2, # in meters
    "aptoffr" : 0., # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 36.5*i2m, # in meters
    "dkoffx": 0.0,  # in meters
    "dkoffy": 0.0,  # in meters
    "eloffx": 0.0,  # in meters
    "eloffz" : 0.0, # in meters
}



# Get the forebaffles and gs out of the way so we can look at a sphere
configDict["test"] = {
    "nbaffles" : 0, # number of forebaffles
    "gsrad" : 1e10, # in meters
    "gsheight" : 0.0001, # in meters
    "gssides" : 8,
    "fbrad": 1e4-1, # in meters
    "fbheight": 1, # in meters
    "winrad" : 0.69/2, # in meters
    "aptoffr" : 0, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 0, # in meters
    "dkoffx": 0.0,  # in meters
    "dkoffy": 0.0,  # in meters
    "eloffx": 0.0,  # in meters
    "eloffz" : 0, # in meters
}

configDict["Custom"] = {
    "nbaffles" : 1, # number of forebaffles
    "gsrad" : 0, # in meters
    #"gsrad" : [2.79,5.05,7.61], # in meters
    "gsheight" : [0,0.94,4.12], # in meters
    "gssides" : 12,
    "fbrad": 0.6, # in meters
    "fbheight": 1.0, # in meters
    "winrad" :0.69/2, # in meters
    "aptoffr" : 1.88/2, # in meters
    "drumangle" : 0, # RADIANS
    "aptoffz" : 36.5*i2m-0.07, # in meters
    "dkoffx": 0.0,  # in meters
    "dkoffy": 0.0,  # in meters
    "eloffx": 0.0,  # in meters
    "eloffz" : 2.3, # in meters
}

configDict["fcolors"] = {
    "groundshield" : 8.675309,
    "fb_main" : 3.264,
    "fb_alts" : 2.9475,
    "floor" : 4.111152,
}

configDict["colors"] = {
    "sky" : 0,
    "fbm" : 1.0,
    "fba" : 2.0,
    "gsh" : 2.0,
    "gnd" : [3.0],
}

configDict["tempcolors"] = {
    "sky" : 0,
    "fbm" : 270,
    "fba" : 270,
    "gsh" : 270,
    "gnd" : [280,270],
}

configDict["BAfff"] = {
    "height" : 3.2, # in meters
    "tilt" : 45., # degrees
    "roll" : 0., # degrees
    "offset" : 0.4, # in meters
    "dims" : [2.75, 1.8], # [width height]
    "color" : 10.0,
}

configDict["keckfff2014"] = {
    "height" : 3.7186, # in meters
    "tilt" : 45., # degrees
    "roll" : 0., # degrees
    #"offset" : 0., # in meters
    "offset" : configDict["keck"]["aptoffr"]-0.191, # Back
    #"offset" : 0.2571-configDict["keck"]["aptoffr"], # fwd
    #"offset" : 0.1047-configDict["keck"]["aptoffr"], # fwdest
    "dims" : [2.75, 1.8], # [width height]
    "color" : 10.0,
}
