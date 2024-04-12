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

typedef struct {
    double *phi;
    double *philocs;
    double *sind_d, *a, *d;
    double *cos_a, *twohav_aod;
    double *e_a1, *e_a2, *e_a3;
    double *np1, *np2, *np3;
    double *npt, *npp;
} KernelLocals;

#endif