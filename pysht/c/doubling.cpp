#include <stddef.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include "doubling.h"
#include "doubling.cuh"

namespace nb = nanobind;
using namespace nb::literals;

intptr_t Cdoubling_ptrs(intptr_t device_pointings_ptr, int nphi, int nring) {
    double* device_pointings = reinterpret_cast<double*>(device_pointings_ptr);
    double* devres = CUdoubling(device_pointings, nphi, nring);
    return reinterpret_cast<intptr_t>(devres);
}

NB_MODULE(dopy, m) {
    m.def(
        "Cdoubling_ptrs",
        [](
            intptr_t device_pointings,
            int nphi,
            int nring) {
            return Cdoubling_ptrs(device_pointings, nphi, nring);
        }
    );
}