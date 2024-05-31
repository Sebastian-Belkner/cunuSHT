# !/bin/bash


######################################################
##########        gclm2lenmap           ##############
######################################################


# for epsilon in 1e-12 1e-10 1e-08 1e-06
# do
#     for i in {1..11}
#     do
#         for n in {01..10}
#         do
#         lmax=$((512*$i-1))
#         fn=/mnt/home/sbelkner/git/cunuSHT/test/benchmark/timings/GPU_cufinufft_transformer/gclm2lenmap/lmax${lmax}_epsilon${epsilon}_run${n}
        
#         SECONDS=0
#             if [ ! -f $fn ]; then
#                 echo "fn does not exist: $fn"
#                 python3 benchmark_gclm2lenmap.py "$i" "$epsilon" "GPU" $n
#                 echo "Finish lmax: $lmax ($i/22), epsilon: $epsilon, run: $n/10 - elapsed time: $SECONDS seconds"
#             else
#                 echo "File already exists: $fn"
#             fi
#         done
#     done
# done


for epsilon in 1e-02 #1e-12 1e-10 1e-08 1e-06
do
    for i in {1..11}
    do
        for n in {01..10}
        do
            value=$((512*$i-1))
            SECONDS=0
            python3 benchmark_gclm2lenmap.py "$i" "$epsilon" "CPU" &> timings/CPU_Lenspyx_transformer/gclm2lenmap/preprocessing/lmax${value}_epsilon${epsilon}_run${n}
            echo "Finish lmax: $value ($i/11), epsilon: $epsilon, run: $n/10 - elapsed time: $SECONDS seconds"
        done
    done
done


######################################################
##########        lenmap2gclm           ##############
######################################################


# for epsilon in 1e-12 1e-10 1e-08 1e-06
# do
#     for i in {1..11}
#     do
#         for n in {01..10}
#         do
#         lmax=$((512*$i-1))
#         fn=/mnt/home/sbelkner/git/cunuSHT/test/benchmark/timings/GPU_cufinufft_transformer/lenmap2gclm/lmax${lmax}_epsilon${epsilon}_run${n}
        
#         SECONDS=0
#             if [ ! -f $fn ]; then
#                 echo "fn does not exist: $fn"
#                 python3 benchmark_lenmap2gclm.py "$i" "$epsilon" "GPU" $n
#                 echo "Finish lmax: $lmax ($i/22), epsilon: $epsilon, run: $n/10 - elapsed time: $SECONDS seconds"
#             else
#                 echo "File already exists: $fn"
#             fi
#         done
#     done
# done


for epsilon in 1e-12 1e-10 1e-08 1e-06 1e-04 1e-02
do
    for i in {1..11}
    do
        for n in {01..10}
        do
            SECONDS=0
            value=$((512*$i-1))
            python3 benchmark_lenmap2gclm.py "$i" "$epsilon" "CPU" &> timings/CPU_Lenspyx_transformer/lenmap2gclm/preprocessing/lmax${value}_epsilon${epsilon}_run${n}
            echo "Finish lmax: $value ($i/11), epsilon: $epsilon, run: $n/10 - elapsed time: $SECONDS seconds"
        done
    done
done