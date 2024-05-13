# !/bin/bash


######################################################
##########        gclm2lenmap           ##############
######################################################


# for i in {1..19}
# do
#     value=$((256*$i-1))
#     python3 benchmark_gclm2lenmap.py "$i" "GPU"
# done

# for i in {1..19}
# do
#     value=$((256*$i-1))
#     python3 benchmark_gclm2lenmap.py "$i" "CPU"  &> timings/CPU_Lenspyx_transformer/gclm2lenmap/preprocessing/lmax${value}_epsilon1e-10
# done




######################################################
##########        lenmap2gclm           ##############
######################################################


# for i in {1..19}
# do
#     value=$((256*$i-1))
#     python3 benchmark_gclm2lenmap.py "$i" "GPU"
# done

for i in {1..19}
do
    value=$((256*$i-1))
    python3 benchmark_lenmap2gclm.py "$i" "GPU"
done