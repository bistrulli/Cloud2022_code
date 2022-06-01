from curses import raw
import os
import time
import shutil
import subprocess
import pandas as pd
import json
import threading

from pathlib import Path

from LoadGenerator import LoadGenerator

# Comment in one of the following experiment settings for the load generator

# For the different workload experiments
#exec(open("settings_diffLoad.py").read())

# For the scaling experiments
#exec(open("settings_scaling.py").read())

loadgenerators = [LoadGenerator(opt_detect), LoadGenerator(opt_fetch)]

data_keys = ["req_id",
            "timestamp",
            "method_call",
            "response_code",
            "duration",
            "duration_upstream",
            "bytes_sent", 
            "bytes_received", 
            "downstream_pod_ip",
            "upstream_cluster",
            "upstream_pod_ip"]

MCsims = 1
gather_time_buffer_before = 10
gather_time_buffer_after = 20
gather_time_restart = 60

# Settings
FRONTEND_NAME = "frontend"
FRONTEND_CLUSTERS = ["cluster-1"]

BACKEND_V1_NAME = "backend-v1"
BACKEND_V1_CLUSTERS = ["cluster-1"]

BACKEND_V2_NAME = "backend-v2"
BACKEND_V2_CLUSTERS = ["cluster-2"]

STORAGE_NAME = "storage"
STORAGE_CLUSTERS = ["cluster-1"]

ALL_NAMES = [FRONTEND_NAME, BACKEND_V1_NAME, BACKEND_V2_NAME, STORAGE_NAME]
ALL_CLUSTERS = [FRONTEND_CLUSTERS, BACKEND_V1_CLUSTERS, BACKEND_V2_CLUSTERS, STORAGE_CLUSTERS]

ROOT = "/home/ubuntu/run_on_gateway/clusters/"

logpath = "logs/"

def log_data(rawdatapath, suffix):
    dataprocs = []
    datafiles = []

    for (i, name) in enumerate(ALL_NAMES):
        for cluster in ALL_CLUSTERS[i]:
            Path(os.path.join(rawdatapath, name, cluster)).mkdir(parents=True, exist_ok=True)
    
    savepath = lambda name, cluster: os.path.join(rawdatapath, name, cluster)
    
    for frontend_cluster in FRONTEND_CLUSTERS:
        arg =   "kubectl --context={}".format(frontend_cluster).split() + \
                "-n facedetect logs -l app=frontend -c istio-proxy -f".split()
        f = os.path.join(savepath(FRONTEND_NAME, frontend_cluster), 
            "frontend-{}-{}.log".format(frontend_cluster, suffix))
        datafiles.append(open(f, "w"))
        dataprocs.append(subprocess.Popen(arg, stdout=datafiles[-1])) 


    for backend_cluster in BACKEND_V1_CLUSTERS:
        arg =   "kubectl --context={}".format(backend_cluster).split() + \
                "-n facedetect logs -l app=backend-v1 -c istio-proxy -f".split()  
        f = os.path.join(savepath(BACKEND_V1_NAME, backend_cluster), 
            "backend-v1-{}-{}.log".format(backend_cluster, suffix))
        datafiles.append(open(f, "w"))
        dataprocs.append(subprocess.Popen(arg, stdout=datafiles[-1])) 

    for backend_cluster in BACKEND_V2_CLUSTERS:
        arg =   "kubectl --context={}".format(backend_cluster).split() + \
                "-n facedetect logs -l app=backend-v2 -c istio-proxy -f".split()  
        f = os.path.join(savepath(BACKEND_V2_NAME, backend_cluster), 
            "backend-v2-{}-{}.log".format(backend_cluster, suffix))
        datafiles.append(open(f, "w"))
        dataprocs.append(subprocess.Popen(arg, stdout=datafiles[-1])) 

    for storage_cluster in STORAGE_CLUSTERS:
        arg =   "kubectl --context={}".format(storage_cluster).split() + \
                "-n facedetect logs -l app=storage -c istio-proxy -f".split()  
        f = os.path.join(savepath(STORAGE_NAME, storage_cluster), 
            "storage-{}-{}.log".format(storage_cluster, suffix))
        datafiles.append(open(f, "w"))
        dataprocs.append(subprocess.Popen(arg, stdout=datafiles[-1]))

    return dataprocs, datafiles

def gen_data(tracepath, rawdatapath):
    for (i, name) in enumerate(ALL_NAMES):
        for cluster in ALL_CLUSTERS[i]:
            dfs = []
            for child in Path(os.path.join(rawdatapath, name, cluster)).iterdir():
                logfile = str(child)
                if logfile.split('.')[-1] != 'log':
                    continue
                
                lines = []
                with open(logfile, "r") as f:
                    lines = f.readlines()

                data = []
                for line in lines:
                    try: 
                        d_tmp = json.loads(line)
                        if set(data_keys) == set(d_tmp.keys()):
                            data.append([d_tmp[key] for key in data_keys])
                    except json.decoder.JSONDecodeError:
                        pass

                dfs.append(pd.DataFrame(data, columns=data_keys))

            df_all = pd.concat(dfs).drop_duplicates().sort_values("timestamp").reset_index(drop=True)
            datafile = os.path.join(tracepath, name + "-" + cluster + ".csv")
            df_all.to_csv(datafile, sep=",", encoding="utf-8")

if __name__ == "__main__":

    if os.path.isdir(logpath):
        shutil.rmtree(logpath)
    os.mkdir(logpath)

    for sim in range(MCsims):
        simpath = os.path.join(logpath, "sim{}".format(sim+1))
        tracepath = os.path.join(simpath, "traces")
        rawdatapath = os.path.join(tracepath, "raw")

        # Start data gathering
        dataprocs, datafiles = log_data(rawdatapath, 1)
        time.sleep(gather_time_buffer_before)

        # Run the load generators
        for lg in loadgenerators:
            lg.sim_id = "{}/{}".format(sim+1, MCsims)

        threads = [threading.Thread(target=lg.run, args=(simpath,)) for lg in loadgenerators]

        [th.start() for th in threads]

        # restart trace gathering after X seconds, since it has some
        # weird timeout issues for longer simulations
        t0 = time.time()
        c = 2
        while any([th.is_alive() for th in threads]):
            time.sleep(0.1)
            if time.time() - t0 > gather_time_restart:
                # start new gathering
                dataprocs_new, datafiles_new = log_data(rawdatapath, c)

                # wait
                time.sleep(gather_time_buffer_before)

                # terminate old
                [dp.terminate() for dp in dataprocs]
                [df.close() for df in datafiles]
                dataprocs = dataprocs_new
                datafiles = datafiles_new

                t0 = time.time()
                c += 1

        [th.join() for th in threads]

        # Wait and terminate after sufficiently long time
        time.sleep(gather_time_buffer_after) 
        [dp.terminate() for dp in dataprocs]
        [df.close() for df in datafiles]

        # Post process data
        gen_data(tracepath, rawdatapath)



