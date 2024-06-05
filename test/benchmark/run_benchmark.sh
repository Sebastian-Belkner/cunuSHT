# !/bin/bash


######################################################
##########        gclm2lenmap           ##############
######################################################


# for epsilon in 1e-12 1e-06 1e-02 # 1e-12 1e-10 1e-08 1e-06 # 1e-12 1e-10 1e-08 1e-06 1e-04 1e-02
# do
#     for run in {01..05}
#     do
#         for ntheta in 36 64 76 136 148 176 244 316 344 376 568 676 736 876 1216 1324 1576 1716 1876 2188 2836 3088 3376 3676 4376 5104 # for nthetai in {1..1}
#         do
#             # ntheta=$((512*$i-1))
#                 SECONDS=0
#                 # fn=/mnt/home/sbelkner/git/cunuSHT/test/benchmark/timings/GPU_cufinufft_transformer/gclm2lenmap/ntheta${ntheta}_epsilon${epsilon}_run${run}
#                 # if [ ! -f $fn ]; then
#                     # echo "fn does not exist: $fn"
#                     python3 benchmark_gclm2lenmap.py "$((ntheta-1))" "$epsilon" "GPU" $run
#                     echo "Finish ntheta: $((ntheta-1)), epsilon: $epsilon, run: $run/10 - elapsed time: $SECONDS seconds"
#                 # else
#                     # echo "File already exists: $fn"
#                 # fi
#         done
#     done
# done

# for epsilon in 1e-12 1e-06 1e-02 # 1e-12 1e-10 1e-08 1e-06 # 1e-12 1e-10 1e-08 1e-06 1e-04 1e-02
# do
#     for run in {01..05}
#     do
#         for ntheta in 36 64 76 136 148 176 244 316 344 376 568 676 736 876 1216 1324 1576 1716 1876 2188 2836 3088 3376 3676 4376 5104 # for nthetai in {1..1}
#         do
# #           # ntheta=$((512*$i-1))
#             SECONDS=0
#             python3 benchmark_gclm2lenmap.py "$((ntheta-1))" "$epsilon" "CPU" &> timings/CPU_Lenspyx_transformer/gclm2lenmap/preprocessing/ntheta$((ntheta-1))_epsilon${epsilon}_run${run}
#             echo "Finish ntheta: $((ntheta-1)), epsilon: $epsilon, run: $run/10 - elapsed time: $SECONDS seconds"
#         done
#     done
# done


######################################################
##########        lenmap2gclm           ##############
######################################################


for epsilon in 1e-12 1e-06 1e-02  # 1e-12 1e-10 1e-08 1e-06 # 1e-12 1e-10 1e-08 1e-06 1e-04 1e-02
do
    for run in {01..02} # {03..05}
    do
        for ntheta in 36 64 76 136 148 176 244 316 344 376 568 676 736 876 1216 1324 1576 1716 1876 2188 2836 3088 3376 3676 4376 5104 # for nthetai in {1..1}
        do
#           # ntheta=$((512*$i-1))
            SECONDS=0
#             # if [ ! -f $fn ]; then
#                 # echo "fn does not exist: $fn"
                python3 benchmark_lenmap2gclm.py "$((ntheta-1))" "$epsilon" "GPU" $run
                echo "Finish ntheta: $((ntheta-1)), epsilon: $epsilon, run: $run/10 - elapsed time: $SECONDS seconds"
#             # else
#                 # echo "File already exists: $fn"
#             # fi
        done
    done
done


for epsilon in 1e-12 1e-06 1e-02  # 1e-12 1e-10 1e-08 1e-06 # 1e-12 1e-10 1e-08 1e-06 1e-04 1e-02
do
    for run in {01..02} # {03..05}
    do
        for ntheta in 36 64 76 136 148 176 244 316 344 376 568 676 736 876 1216 1324 1576 1716 1876 2188 2836 3088 3376 3676 4376 5104 # for nthetai in {1..1}
        do
#           # ntheta=$((512*$i-1))
            SECONDS=0
            python3 benchmark_lenmap2gclm.py "$((ntheta-1))" "$epsilon" "CPU" &> timings/CPU_Lenspyx_transformer/lenmap2gclm/preprocessing/ntheta$((ntheta-1))_epsilon${epsilon}_run${run}
            echo "Finish ntheta: $((ntheta-1)), epsilon: $epsilon, run: $run/10 - elapsed time: $SECONDS seconds"
        done
    done
done