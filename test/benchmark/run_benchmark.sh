#!/bin/bash
for i in {1..21}
do
    value=$((256*$i-1))
    python3 benchmark_gclm2lenmap.py "$i"
    # python3 benchmark_gclm2lenmap.py "$value"
done


# for i in {1..24}
# do
#     value=$((256*$i-1))
#     python3 benchmark_lenmap2gclm.py "$value"
# done