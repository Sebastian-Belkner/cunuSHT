// pointing.cuh
#ifndef POINTING_CUH
#define POINTING_CUH


void CUpointing_DUCC(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result); // Declare the CUDA kernel function

void CUpointing_lenspyx(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result); // Declare the CUDA kernel function


#endif // POINTING_CUH