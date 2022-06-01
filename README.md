### Reproducing the experimental evaluations

The code in this repository can be used to reproduce the experimental evaluations seen in the paper

> Johan Ruuskanen, Anton Cervin, "Distributed online extraction of a fluid model for microservice applications using local tracing data", to appear at IEEE CLOUD 2022

The experiments are run on the [FedApp sandbox](https://github.com/JohanRuuskanen/FedApp), while the models are extracted and evaluated using `julia-1.7.0`.

In order to recreate the figures seen in the paper, you must first deploy the sandbox with the example application, run the experiments and finally extract the models. 



#### Deploy the FedApp sandbox

Access to an OpenStack cloud is required to deploy the sandbox out-of-the-box.  The sandbox can possibly also be deployed on other infrastructures,  but this requires some tinkering. 

Follow the steps in shown in the [sanbox repo](https://github.com/JohanRuuskanen/FedApp), and deploy the gateway with  4vCPU and 16 Gb of RAM, along with 2 cluster each with 4 virtual machines, each with 4vCPU and8 Gb of RAM. 

Finally, add a delay of 25 ms between the two clusters using TC netem.



#### Application deployment

To deploy the example application on the FedApp sandbox, begin by copying  the `application/facedetect_3s` folder to the gateway.

We used the [UMass face detection database](<http://vis-www.cs.umass.edu/fddb/> )  to provide the necessary images for loading the application. Download it and extract it to the `application/data` folder on the gateway.

To avoid time-dependencies of requests fetching images not yet stored, we pre-populated the storage service with the possible images fetched. To do this, simply run `application/facedetect_3s/apps/storage/populate.py` before proceeding.

In `application/facedetect_3s/apps`, change the gitlab repository in the `build_and_push.sh` script to point to a gitlab container registry that you can access.  Then run the following script to build and push all necessary container images.

```
chmod +x build_and_push.sh
./build_and_push.sh
```

Also, change the gitlab repository in each of the 4 service YAMLs

```
backend-v1.yaml
backend-v2.yaml
frontend.yaml
storage.yaml
```

The application can then be deploy with

```
python3 deploy.py
```

To later remove it, simply run 

```
python3 remove.py
```

The application can be reached on port 3001 on the gateway, and to access it simply perform a port-forward using e.g. ssh `ssh -L 3001:localhost:3001 ubuntu@GATEWAY_IP` and then visit `http://localhost:3001` on your local computer. 




#### Running the experiments


After the application has been deployed, visit the `application/facedetect_3s/load_generation` folder. In `run_experiment.py`, choose which experiment to run, and then run it with

```
python3 run_experiment.py
```

The generated logs in `application/facedetect_3s/load_generation/logs` contains the necessary tracing data for the model extraction.

After running the workload and scaling experiments, copy the gathered `logs/` folders to the `model_extracion` folder on your local computer. 



#### Predicting with the smoothed fluid model

The model extraction and prediction experiments can be performed in the `model_extraction` folder. They are implemented in `Julia` and  tested with `Julia-1.7.0`. For the plotting, it further requires `Python` with `matplotlib`.

To extract a model and perform predictions, start `Julia` in the  `model_extraction` folder and type

```julia
] activate .
] instantiate
```

to setup the environment.

The workload experiments are found in `examine_model_diffLoad.jl`, while the scaling experiments are found in `examine_model_scaling.jl`. Simply change the `datafolder = ""` in the two experiment scripts to point to the correct tracing log folder from the experiments and run the script.