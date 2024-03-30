#include <nanobind/nanobind.h>


#include "pointing.h"
#include "pointing.cuh"


void Cpointing_lenspyx(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    CUpointing_lenspyx(thetas_, phi0_, nphis, ringstarts, red, imd, nrings, npix, host_result);
}


void Cpointing_DUCC(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result) {
    CUpointing_DUCC(thetas_, phi0_, nphis, ringstarts, red, imd, nrings, npix, host_result);
}


NB_MODULE(pointing, m) {
    m.def("Cpointing_lenspyx", &Cpointing_lenspyx);
    m.def("Cpointing_DUCC", &Cpointing_DUCC);
}