# !/bin/bash
# for i in {10..20}
for i in {1..19}
# for i in {19..19}
do
    value=$((256*$i-1))
    python3 benchmark_gclm2lenmap.py "$i" &> timings/CPU_Lenspyx_transformer/gclm2lenmap/preprocessing/lmax${value}_epsilon0.0001
    # python3 benchmark_gclm2lenmap.py "$value"
done


# for i in {1..24}
# do
#     value=$((256*$i-1))
#     python3 benchmark_lenmap2gclm.py "$value"
# done