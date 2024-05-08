#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/tuple.h>

#include "pointing.cuh"

NB_MODULE(popy, m) {
    m.def("CUpointing_1Dto1D",
        &CUpointing_1Dto1D<double>,
        "thetas"_a.noconvert(),
        "phi0"_a.noconvert(),
        "nphis"_a.noconvert(),
        "ringstarts"_a.noconvert(),
        "spin1_theta"_a.noconvert(),
        "spin1_phi"_a.noconvert(),
        "outarr_pt"_a.noconvert(),
        "outarr_pp"_a.noconvert()
    );
    // m.def("CUpointing_cparr",
    //     &CUpointing_cparr<double>,
    //     "thetas"_a.noconvert(),
    //     "phi0"_a.noconvert(),
    //     "nphis"_a.noconvert(),
    //     "ringstarts"_a.noconvert(),
    //     "synthmap"_a.noconvert(),
    //     "outarr_pt"_a.noconvert(),
    //     "outarr_pp"_a.noconvert()
    // );
}