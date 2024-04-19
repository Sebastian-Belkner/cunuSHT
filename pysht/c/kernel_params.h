#ifndef KERNEL_PARAMS_H
#define KERNEL_PARAMS_H

typedef struct {
    double *thetas;
    double *phi0;
    int *nphis;
    int *ringstarts;
    double *synthmap;
    int nring;
    int npix;
} KernelParams;

typedef struct {
    double *sind_d;
    double *dphi;
    double *thtp;
    double *e_d, *e_t, *e_tp;
    double *d;
    double *philocs;
} KernelLocals_lp;

template <typename Scalar>
struct KernelLocals{
    Scalar *phi;
    Scalar *philocs;
    Scalar *sind_d, *a, *d;
    Scalar *cos_a, *twohav_aod;
    Scalar *e_a1, *e_a2, *e_a3;
    Scalar *np1, *np2, *np3;
    Scalar *npt, *npp;
};

#endif