#!/bin/sh

# usage
SCRIPT_FILE=`basename $0`
function usage()
{
  echo "usage: ${SCRIPT_FILE} DEF_FILE" 1>&2
}

# arguments
if [ $# -ne 1 ]; then
  usage
  exit 1
fi
DEF_FILE=$1

# constant
SCRIPT_DIR=$(cd $(dirname $0); pwd)
LOG_FILE=${SCRIPT_DIR}/../logs/crawling.log
DOWNLOAD_DIR=${SCRIPT_DIR}/../data/images

# log
function log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo "$(date '+%Y-%m-%dT%H:%M:%S') ${PROCNAME} (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $@" | tee -a ${LOG_FILE}
}

# main 
log "info start script"

mkdir -p ${DOWNLOAD_DIR}

cat ${DEF_FILE} | while read URL
do
  cd ${DOWNLOAD_DIR} && { curl -O ${URL} ; cd -; }
  if [ $? -gt 0 ]; then
    log "error download"
  fi
done

log "info end script"
