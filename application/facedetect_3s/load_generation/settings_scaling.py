
import os
import random
from glob import glob
import numpy as np

from LoadGenerator import LoadGeneratorOptions

simSecDur = [300 for _ in range(11)]

# Settings for the detect/ load generator
opt_detect = LoadGeneratorOptions()
opt_detect.url = "http://10.0.1.x:31111/detect/" # Change x to match the IP of the master node in cluster 1
opt_detect.setID("ir")
opt_detect.logFile = "load_generator_ir.csv"
opt_detect.settingsFile = "settings_ir.yaml"
opt_detect.loadType = "open" 
opt_detect.connections = 0
opt_detect.simTime = sum(simSecDur)
opt_detect.printout = True

# Nested function for generating either the wait-time or interarrival time.
def interarrivalTimes(simSecDur):
    simIntEnd = np.cumsum(simSecDur)
    ratios = [1/13 for _ in range(11)]
    def timefunc(t):
        i = 0
        for ssd in simIntEnd[:-1]:
            if t < ssd:
                break
            i += 1
        return np.random.exponential(ratios[i])
    return timefunc
opt_detect.genTimeFunc = interarrivalTimes(simSecDur)

# Nested function for creating a function closure for the data to send
def img_func(datapath):
    image_paths = []
    for dir,_,_ in os.walk(datapath):
        image_paths.extend(glob(os.path.join(dir, "*.jpg")))
    def rand_img_func(t):
        return  {'imgfile': open(image_paths[random.randint(0, len(image_paths)-1)], "rb")}
    return rand_img_func
opt_detect.dataToSend = img_func("../../data/")

# Nested function for processing after the data has sent in the request
def postprocessing():
    def ppfunc(D):
        return D['imgfile'].close()
    return ppfunc
opt_detect.postProcessing = postprocessing()

# Nested function for creating a function closure for additional headers
def headers_func(simSecDur):
    simIntEnd = np.cumsum(simSecDur)
    lbWeights = [
        "1.0,0.0", 
        "0.9,0.1", 
        "0.8,0.2", 
        "0.7,0.3", 
        "0.6,0.4", 
        "0.5,0.5", 
        "0.4,0.6", 
        "0.3,0.7", 
        "0.2,0.8",
        "0.1,0.9",  
        "0.0,1.0"
    ]
    def get_headers(t):
        i = 0
        for ssd in simIntEnd[:-1]:
            if t < ssd:
                break
            i += 1
        return {"lb-weights": lbWeights[i],
                "upstream-timeout": "300.0",
                "storage-extraload": "40"
            }
    return get_headers
opt_detect.headersAdditional = headers_func(simSecDur)
opt_detect.exportDict = {
    "simSecDur": simSecDur
}

# Settings for the fetch/ load generator
opt_fetch = LoadGeneratorOptions()
opt_fetch.url = "http://10.0.1.x:31111/fetch/" # Change x to match the IP of the master node in cluster 1
opt_fetch.setID("st")
opt_fetch.logFile = "load_generator_st.csv"
opt_fetch.settingsFile = "settings_st.yaml"
opt_fetch.loadType = "closed"
opt_fetch.connections = 50
opt_fetch.simTime = sum(simSecDur)
opt_fetch.printout = True
opt_fetch.headersAdditional = lambda t: \
     {"upstream-timeout": "300.0", "storage-extraload": "40"}

# Nested function for generating either the wait-time or interarrival time.
def interarrivalTimes_2(simSecDur):
    simIntEnd = np.cumsum(simSecDur)
    ratios = [1/1.4 for _ in range(11)]
    def timefunc(t):
        i = 0
        for ssd in simIntEnd[:-1]:
            if t < ssd:
                break
            i += 1
        return np.random.exponential(ratios[i])
    return timefunc
opt_fetch.genTimeFunc = interarrivalTimes_2(simSecDur)

# Nested function for creating a function closure for the data to send
def img_func_storage(datapath):
    image_paths = []
    for dir,_,_ in os.walk(datapath):
        image_paths.extend(glob(os.path.join(dir, "*.jpg")))
    def rand_img_func(t):
        filename = image_paths[random.randint(0, len(image_paths)-1)]
        return  {'imgfile': filename.split("/")[-1]}
    return rand_img_func
opt_fetch.dataToSend = img_func_storage("../../data/")
opt_fetch.exportDict = {
    "simSecDur": simSecDur
}
