#!/bin/bash
set -e
find /scratch/preframr/training-dumps/ -name "*.[0-9]*.[0-9]*.parquet" -ls -delete
find /scratch/preframr/training-dumps/ -name "*.uni.zst" -ls -delete
find /scratch/preframr/training-dumps/ -name "*.npy" -ls -delete
