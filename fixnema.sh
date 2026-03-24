#!/bin/bash

set -e

ubxstatus=$(ubxtool -p CFG-PRT|grep -E "outProtoMask.+NEMA")
if [ "${ubxstatus}" == "" ] ; then
  exit 0
fi
ubxtool -e NEMA
