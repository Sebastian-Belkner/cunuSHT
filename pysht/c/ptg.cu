__device__ double ptg(int l, int m, double x) {
    return 0;
}

__global__ void compute_ptg(int lmax, int mmax, int size_x, double *x, double *ptgs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size_x) {
        double x_ = x[threadIdx.x];
        for ir in np.argsort(self.geom.ofs) { // We must follow the ordering of scarf position-space map
            pixs = Geom.rings2pix(self.geom, [ir])
            if (pixs.size > 0) {
                t_red = red[pixs];
                i_imd = imd[pixs];
                phis = Geom.phis(self.geom, ir)[pixs - self.geom.ofs[ir]]
                // assert phis.size == pixs.size, (phis.size, pixs.size)
                thts = self.geom.theta[ir] * np.ones(pixs.size)
                thtp_, phip_ = d2ang(t_red, i_imd, thts , phis, int(np.round(np.cos(self.geom.theta[ir]))))
                sli = slice(startpix, startpix + len(pixs))
                thp_phip_gamma[0, sli] = thtp_
                thp_phip_gamma[1, sli] = phip_
                cot = np.cos(self.geom.theta[ir]) / np.sin(self.geom.theta[ir])
                d = np.sqrt(t_red ** 2 + i_imd ** 2)
                thp_phip_gamma[2, sli] = np.arctan2(i_imd, t_red ) - np.arctan2(i_imd, d * np.sin(d) * cot + t_red * np.cos(d))
                startpix += len(pixs)
            }
            ptgs[idx] = ptg(lmax, mmax, x_);
        }
    }
}

extern "C" void ptg(double* tht, double* phi0, double *nphi, double *ringstart, double *synth_spin1_map) {
    double *device_x, *device_result;

    npix = sum(nphi)
    thp_phip_gamma = np.empty((3, npix), dtype=float)  // (-1) gamma in last arguement
    startpix = 0
    assert(startpix == npix);

    float* theta = self.geom.theta;
    int size = self.geom.theta.size();
    float pi = 3.14159265359;
    bool condition1 = allGreaterThanZero(theta, size);
    bool condition2 = allLessThanPi(theta, size, pi);
    assert(condition1 && condition2);

    red, imd = synth_spin1_map
    // const int threadsPerBlock = 256;
    // const int blocksPerGrid = (lmax + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&device_x, size_x * sizeof(double));
    cudaMalloc((void**)&device_result, size_x * sizeof(double));

    cudaMemcpy(device_x, host_x, size_x * sizeof(double), cudaMemcpyHostToDevice);

    // auto start = high_resolution_clock::now();
    compute_ptg<<<1, size_x>>>(lmax, mmax, size_x, device_x, device_result);
    cudaDeviceSynchronize();
    // auto stop = high_resolution_clock::now();

    cudaMemcpy(host_result, device_result, size_x * sizeof(double), cudaMemcpyDeviceToHost);

    // for (int l = lmax; l <= lmax; ++l) {
        // for (int m = 0; m <= l; ++m) {

    // printf("P_%d^%d(x): ",lmax,mmax);
    // // printf("%d", size_x);
    // for (int i = 0; i < size_x; ++i) {
    //     printf("%.2f ", host_result[i]);
    // }
    // printf("\n");
        // }
    // }

    cudaFree(device_x);
    cudaFree(device_result);
}

bool allGreaterThanZero(float* array, int size) {
    for (int i = 0; i < size; i++) {
        if (array[i] <= 0.0) {
            return false;
        }
    }
    return true;
}

bool allLessThanPi(float* array, int size, float pi) {
    for (int i = 0; i < size; i++) {
        if (array[i] >= pi) {
            return false;
        }
    }
    return true;
}