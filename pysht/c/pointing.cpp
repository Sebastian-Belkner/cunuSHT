#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include "pointing.h"
#include "pointing.cuh"

namespace nb = nanobind;

using namespace nb::literals;

void Cpointing_lenspyx(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    CUpointing_lenspyx(thetas_, phi0_, nphis, ringstarts, red, imd, nrings, npix, host_result);
}


void Cpointing_DUCC(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    CUpointing_DUCC(thetas_, phi0_, nphis, ringstarts, red, imd, nrings, npix, host_result);
}

NB_MODULE(popy, m) {
    // m.def("Cpointing_lenspyx", &Cpointing_lenspyx);
    m.def(
        "Cpointing_DUCC",
        [](
            nb::ndarray<float>& thetas_,
            nb::ndarray<float>& phi0_,
            nb::ndarray<int>& nphis,
            nb::ndarray<int>& ringstarts,
            nb::ndarray<double>& red,
            nb::ndarray<double>& imd,
            int nrings,
            int npix,
            nb::ndarray<double>&host_result) {
            Cpointing_DUCC(thetas_.data(), phi0_.data(), nphis.data(), ringstarts.data(), red.data(), imd.data(), nrings, npix, host_result.data());
        }
    );
}

// NB_MODULE(popy, m) {
//     // m.def("Cpointing_lenspyx", &Cpointing_lenspyx);
//     m.def("Cpointing_DUCC",
//     &Cpointing_DUCC);
// }