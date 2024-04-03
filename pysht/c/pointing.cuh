// pointing.cuh
#ifndef POINTING_CUH
#define POINTING_CUH
#include <tuple>
#include "kernel_params.h"


double* CUpointing_struct(KernelParams hostkp);

// void CUpointing(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nring, int npix, double *host_result); // Declare the CUDA kernel function
// double* CUpointing(float* thetas_, float* phi0_, int* nphis, int* ringstarts, int red, int imd, int nring, int npix, double* device_result, double *host_result); // Declare the CUDA kernel function
// std::tuple<KernelParams*, double*> CUpointing_MemcpyHostToDevice(KernelParams *kp, int nring, int npix);
// int CUpointing_exec(KernelParams *kp, double *device_result, int nring, int npix);
// double* CUpointing_exec_ptrs(int thetas_, int phi0_, int nphis, int ringstarts, int device_red_ptr, int device_imd_ptr, int nring, int npix, double* host_result);
// double* CUpointing_MemcpyDeviceToHost(double* device_result, double *host_result, int npix);
// void dealloacte(double* dev_arr);
// void CUpointing_alt1(float* thetas_, float* phi0_, int* nphis, int* ringstarts, double *red, double *imd, int nring, int npix, double *host_result); // Declare the CUDA kernel function
// double* CUpointing_gpuarray(float* thetas_, float* phi0_, int* nphis, int* ringstarts, int device_red_ptr, int c, int nring, int npix, double *device_result, double *host_result);

#endif // POINTING_CUH