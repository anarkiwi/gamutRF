#!/bin/bash

set -e

VOL_PREFIX=/scratch/gamutrf docker compose -f orchestrator.yml down
rm -rf /scratch/gamutrf/samples/*
VOL_PREFIX=/scratch/gamutrf docker compose -f orchestrator.yml up -d
while [[ "$(find /scratch/gamutrf/samples -type f -name \*meta)" == "" ]] ; do
    find /scratch/gamutrf/samples -type f -ls
    sleep 1
done
VOL_PREFIX=/scratch/gamutrf docker compose -f orchestrator.yml down
/scratch/anarkiwi/rfml/sigmf_freq_demux.py $(find /scratch/gamutrf/samples -type f -name \*meta)
