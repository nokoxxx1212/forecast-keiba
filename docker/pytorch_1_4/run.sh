#!/bin/bash

# usage
SCRIPT_FILE=`basename $0`
function usage()
{
  echo "usage: ${SCRIPT_FILE} docker/nvidia-docker" 1>&2
}

# arguments
if [ $# -ne 1 ]; then
  usage
  exit 1
fi
DOCKER_CMD=$1

# constant
CONTAINER_NAME=forecast-keiba
CONTAINER_IMAGE=forecast-keiba
HOST_PORT_IPYNB=28888
CONTAINER_PORT_IPYNB=8888
HOST_PORT_KV=24141
CONTAINER_PORT_KV=4141
HOST_PORT_MLFLOW=25000
CONTAINER_PORT_MLFLOW=5000

# main
${DOCKER_CMD} run --name ${CONTAINER_NAME} \
  --privileged \
  --entrypoint bash \
  -v $(pwd)/../../:/opt/forecast-keiba/ \
  -p ${HOST_PORT_IPYNB}:${CONTAINER_PORT_IPYNB} \
  -p ${HOST_PORT_KV}:${CONTAINER_PORT_KV} \
  -p ${HOST_PORT_MLFLOW}:${CONTAINER_PORT_MLFLOW} \
  -it ${CONTAINER_IMAGE}
