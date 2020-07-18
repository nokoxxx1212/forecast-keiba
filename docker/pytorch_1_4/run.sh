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
HOST_PORT=28888
CONTAINER_PORT=8888

# main
${DOCKER_CMD} run --name ${CONTAINER_NAME} \
  --privileged \
  --entrypoint bash \
  -v $(pwd)/../../:/opt/forecast-keiba/ \
  -p ${HOST_PORT}:${CONTAINER_PORT} \
  -it ${CONTAINER_IMAGE}
