#!/bin/bash
for i in {1..20}
do
    value=$((256*$i-1))
    python3 benchmark_gclm2lenmap.py "$1" "$value"
done