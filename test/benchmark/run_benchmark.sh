#!/bin/bash
for i in {8..13}
do
    value=$((256*$i-1))
    python3 benchmark_gclm2lenmap.py "$1" "$value"
done