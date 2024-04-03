// pointing.cuh
#ifndef POINTING_CUH
#define POINTING_CUH
#include <tuple>
#include "kernel_params.h"

struct Params {
    double *red;
    double *imd;
    int nring;
    int npix;
};

void CUpointing_devres3(int npix, double* devres);
void CUpointing_arrdevres3(double *arr, int nring, int npix, double* devres);

double* CUpointing_devres11(int npix);
void CUpointing_devres12(int npix, double* devres);
void CUpointing_devres13(int npix, double* devres, double* host_result);

void CUpointing_garrdevres3(double *thetas, int nring, int npix, double* host_result);

void CUpointing_structdevres3(Params params, double* host_result);

std::tuple<Params,double*> CUpointing_structdevres11(Params params);
void CUpointing_structdevres12(Params hostparams, std::tuple<Params,double*> devtup);
void CUpointing_structdevres13(Params hostparams, double* devres, double* host_result);

double* CUpointing_structdevres3_retptr(Params params);

#endif // POINTING_CUH