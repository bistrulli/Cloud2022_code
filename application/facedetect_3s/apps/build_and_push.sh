#!/usr/bin/env bash

export GITLAB_REGISTRY=hub.docker.com # Insert git container registry, we used gitlab
export REGPATH=bistrulli # Path to the registry

docker login $GITLAB_REGISTRY

for d in */ ; do
  docker build -t $GITLAB_REGISTRY/$REGPATH/"${d///}" $d.
  docker push $GITLAB_REGISTRY/$REGPATH/"${d///}"
done
