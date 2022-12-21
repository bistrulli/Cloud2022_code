#!/usr/bin/env bash

export GITLAB_REGISTRY=hub.docker.com # Insert git container registry, we used gitlab
export REGPATH=bistrulli # Path to the registry

docker login

for d in */ ; do
  docker build -t $REGPATH/"${d///}" $d.
  docker push $REGPATH/"${d///}"
done
