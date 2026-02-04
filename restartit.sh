#!/bin/sh
set -e
cd $(dirname $0)/gamutrf
export VOL_PREFIX=/scratch/gamutrf 
export NEST=$(hostname)
docker compose -f orchestrator.yml -f torchserve-orin.yml down
docker compose -f orchestrator.yml -f torchserve-orin.yml up -d torchserve gamutrf
