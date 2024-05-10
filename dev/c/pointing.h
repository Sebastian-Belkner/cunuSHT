#ifndef POINTING_H
#define POINTING_H
#include "kernel_params.h"
namespace nb = nanobind;
using namespace nb::literals;

#include <nanobind/stl/tuple.h>

std::tuple<intptr_t, intptr_t> Cpointing_ptrs(intptr_t thetas_, intptr_t phi0_, intptr_t nphis_, intptr_t ringstarts_, intptr_t synthmap_, int nring, int npix, double *host_result);

#endif  // POINTING_H