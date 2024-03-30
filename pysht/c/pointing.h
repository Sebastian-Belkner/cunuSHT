
#ifndef POINTING_DUCC_H
#define POINTING_DUCC_H

#ifdef __cplusplus
extern "C" {
#endif

void Cpointing_lenspyx(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result);

void Cpointing_DUCC(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nrings, int npix, double *host_result);

#ifdef __cplusplus
}
#endif

#endif  // POINTING_DUCC_H