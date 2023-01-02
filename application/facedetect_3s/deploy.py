import os
import yaml
import time
import os.path

# Settings
FRONTEND_NAME = "frontend"
FRONTEND_CLUSTERS = ["minikube"]

BACKEND_V1_NAME = "backend-v1"
BACKEND_V1_CLUSTERS = ["minikube"]

BACKEND_V2_NAME = "backend-v2"
BACKEND_V2_CLUSTERS = ["minikube"]

STORAGE_NAME = "storage"
STORAGE_CLUSTERS = ["minikube"]

ROOT = "/home/ubuntu/run_on_gateway/clusters/"

def deploy(clusters):
    for cluster in clusters:
        print("====== {} ======".format(cluster))
        if not ((cluster in FRONTEND_CLUSTERS) or (cluster in BACKEND_V1_CLUSTERS) \
                or (cluster in BACKEND_V2_CLUSTERS)):
            continue
        os.system("kubectl --context={} apply -f src/namespace.yaml".format(cluster))
        #os.system("kubectl --context={} -n facedetect")#create secret generic gitlab-auth".format(cluster) \
            #+ " --from-file=.dockerconfigjson=%s/.docker/config.json"%(os.path.expanduser("~")) \
            #+ " --type=kubernetes.io/dockerconfigjson")
        os.system("kubectl --context={} apply -f src/backend-v1.yaml -l service=backend-v1".format(cluster))
        os.system("kubectl --context={} apply -f src/backend-v2.yaml -l service=backend-v2".format(cluster))
        os.system("kubectl --context={} apply -f src/storage.yaml -l service=storage".format(cluster))


def deploy_frontend():
    for frontend_cluster in FRONTEND_CLUSTERS:
        print("====== {} ======".format(frontend_cluster))
        os.system("kubectl --context={} apply -f src/frontend.yaml".format(frontend_cluster))

def deploy_backend():
    for backend_cluster in BACKEND_V1_CLUSTERS:
        print("====== {} ======".format(backend_cluster))
        os.system("kubectl --context={} apply -f src/backend-v1.yaml".format(backend_cluster))
    for backend_cluster in BACKEND_V2_CLUSTERS:
        print("====== {} ======".format(backend_cluster))
        os.system("kubectl --context={} apply -f src/backend-v2.yaml".format(backend_cluster))

def deploy_storage():
    for storage_cluster in STORAGE_CLUSTERS:
        print("====== {} ======".format(storage_cluster))
        os.system("kubectl --context={} apply -f src/storage.yaml".format(storage_cluster))

def retrieve_cluster_IPs(clusters):
    ips = {}
    for (i, cluster) in enumerate(clusters):
        
        role="master"
        if(len(clusters)==1):
            role="control-plane"
        
        CMD = "kubectl --context=%s get nodes -o wide | grep %s | awk '{print $6}'" % (cluster,role)
        masterip = os.popen(CMD).read().strip()
        ips[cluster] = masterip
    return ips

def generate_nginxconf(clusters):

    ips = retrieve_cluster_IPs(clusters)

    us_imgrec = ""
    for frontend_cluster in FRONTEND_CLUSTERS:
        us_imgrec += "server %s:31111;\n" % ips[frontend_cluster]

    conf = """
    user    nginx;
    worker_processes        1;
    error_log  /var/log/nginx/error.log warn;
    pid        /var/run/nginx.pid;

    events {
            worker_connections      1024;
    }
    http {
            proxy_http_version 1.1;

            client_max_body_size 12M;

            upstream us_imgrec {
                %s
            }

            server {
                    listen 3001;

                    location / {
                            proxy_pass http://us_imgrec/;
                    }

                    location /detect/ {
                            proxy_pass http://us_imgrec/detect/;
                    }

                    location /fetch/ {
                            proxy_pass http://us_imgrec/fetch/;
                    }
            }
    }
    """ % (us_imgrec)

    with open("nginx.conf", "w") as f:
        f.write(conf)


def launch_nginx():
    os.system("docker run -d  --network=host \
        -v %s/nginx.conf:/etc/nginx/nginx.conf:ro \
        --name=nginx_loadbalancer nginx:1.17.10" % (os.getcwd()))

if __name__ == '__main__':

    #clusters = [dir for dir in os.listdir(ROOT) if "cluster-" in dir]
    clusters=["minikube"]
    clusters.sort()

    deploy(clusters)
    #deploy_backend()
    #deploy_storage()
    #deploy_frontend()
    generate_nginxconf(clusters)
    launch_nginx()
