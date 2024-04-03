// pointing.cuh
#ifndef POINTING_CUH
#define POINTING_CUH
#include <tuple>
#include "kernel_params.h"


std::tuple<double*, double*> CUpointing_struct(KernelParams hostkp);

#endif // POINTING_CUH