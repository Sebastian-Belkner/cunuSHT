## profile


to run profiling:
    add decorator @profile to the function
    run kernprof -l -v script.py
    to run the script with the profiler
    python -m cProfile -o output.pstats script.py
    to visualize the output
    snakeviz output.pstats
    or
    pyprof2calltree -i output.pstats -k
    

To profile:
 1. `kernprof -l gclm2lenmap_GPU.py'`
 2. `python3 -m line_profiler -rmt "gclm2lenmap_GPU.py.lprof"

