#ifndef POINTING_CUH
#define POINTING_CUH

#include "kernel_params.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

namespace nb = nanobind;
using namespace nb::literals;

template void CUpointing_1Dto1D<double>(
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> thetas,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> phi0,
    nb::ndarray<const size_t, nb::ndim<1>, nb::device::cuda> nphis,
    nb::ndarray<const size_t, nb::ndim<1>, nb::device::cuda> ringstarts,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> spin1_theta,
    nb::ndarray<const Scalar, nb::ndim<1>, nb::device::cuda> spin1_phi,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_pt,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_pp);

#endif // POINTING_CUH