#ifndef POINTING_H
#define POINTING_H
#include "kernel_params.h"

intptr_t Cpointing_ptrs(intptr_t thetas_, intptr_t phi0_, intptr_t nphis, intptr_t ringstarts, intptr_t device_red_ptr, intptr_t device_imd_ptr, int nring, int npix, double *host_result);
// void Cpointing_ptrs(float* thetas_, float* phi0_, int* nphis, int* ringstarts, int red, int imd, int nring, int npix, double *host_result);
// int Cpointing_gpuarray(float* thetas_, float* phi0_, int* nphis, int* ringstarts, int red, int imd, int nring, int npix, double *host_result);
// int Cpointing_exec(KernelParams *kp, double* dev_result, int nring);
// double* Cpointing_MemcpyDeviceToHost(int device_memaddress, double *host_result, int npix);
// double* Cpointing_MemcpyHostToDevice(KernelParams *kp, double* thetas, double* phi0, int* nphis, int* ringstarts, int nring, int npix);
// void dealloacte(double* dev_arr);
// void Cpointing_alt1(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nring, int npix, double *host_result);
// void float_to_double(const float* src, double* dest, int size);
#endif  // POINTING_H