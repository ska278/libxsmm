#!/bin/bash

HERE=$(cd $(dirname $0); pwd -P)
GREP=$(which grep)
ENV=$(which env)

if [[ "Windows_NT" == "${OS}" ]] ; then
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(which cygcheck)
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../lib
else
  LDD=$(which ldd)
fi

echo "============="
echo "Running tests"
echo "============="
for TEST in $(ls -1 ${HERE}/*.c) ; do
  NAME=$(basename ${TEST} .c)
  echo -n "${NAME}... "

  ERROR=$({
  if [[ "-mic" != "$1" ]] ; then
    if [[ "" != "$(${LDD} ${HERE}/${NAME} | ${GREP} libiomp5\.so)" ]] ; then
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        KMP_AFFINITY=scatter,granularity=fine,1 \
        MIC_KMP_AFFINITY=scatter,granularity=fine \
        MIC_ENV_PREFIX=MIC \
        OFFLOAD_INIT=on_start \
      ${HERE}/${NAME} $*
    else
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        OMP_PROC_BIND=TRUE \
      ${HERE}/${NAME} $*
    fi
  else
    shift
    ${ENV} \
      SINK_LD_LIBRARY_PATH=${SINK_LD_LIBRARY_PATH}:${MIC_LD_LIBRARY_PATH}:${HERE}/../lib/mic \
    micnativeloadex \
      ${HERE}/${NAME} -a "$*" \
      -e "KMP_AFFINITY=scatter,granularity=fine"
  fi > /dev/null; } 2>&1)
  if [[ 0 != $? ]] ; then
    echo "FAILED $ERROR"
    exit 1
  else
    echo "OK"
  fi
done

