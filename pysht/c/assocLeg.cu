#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>
#include <complex>

using namespace std::chrono;


// double* linspace(double start, double stop, int num_points) {
//     double* result = new double[num_points];
//     double step = (stop - start) / (num_points - 1);
//     for (int i = 0; i < num_points; ++i) {
//         result[i] = start + i * step;
//     }
//     return result;
// }

__device__ double dsqrt(double n) {
    return sqrt(n);
}
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// std::vector<double> vectorizedCosine(const std::vector<double>& input) {
//     std::vector<double> result;
//     result.reserve(input.size());

//     for (double val : input) {
//         result.push_back(std::cos(val));
//     }

//     return result;
// }
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> meshgrid(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<std::vector<double>> X(x.size(), std::vector<double>(y.size()));
    std::vector<std::vector<double>> Y(x.size(), std::vector<double>(y.size()));

    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            X[i][j] = x[i];
            Y[i][j] = y[j];
        }
    }
    return std::make_pair(X, Y);
}
std::pair<std::vector<double>, std::vector<double>> meshgrid_1d(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double>X(x.size()*y.size());
    std::vector<double>Y(x.size()*y.size());

    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            X[i*x.size()+j] = x[j];
            Y[i*x.size()+j] = y[i];
        }
    }
    return std::make_pair(X, Y);
}

__device__ double factorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}
__device__ double power(double base, int exponent) {
    double result = 1.0;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}
__device__ double normalization_constant(int l, int m) {
    return dsqrt((2 * l + 1) * factorial(l - m) / (4 * M_PI * factorial(l + m)));
}
__device__ double legendre(int l, int m, double x) {
    double fact = factorial(l - m);
    double Pmm = power(-1, m) * fact * power(1.0 - x * x, m / 2.0);
    double Pm_minus_1 = power(-1, m) * (2.0 * m - 1.0) * power(1.0 - x * x, m / 2.0 - 1.0);
    if (l == m) {
        return Pmm;
    }
    double Pm_plus_1 = power(-1, m) * (2.0 * m + 1.0) * x * Pm_minus_1;
    if (l == m + 1) {
        return Pm_plus_1;
    }
    double Plm;
    for (int i = m + 2; i <= l; ++i) {
        Plm = (x * (2.0 * i - 1.0) * Pm_plus_1 - (i + m - 1.0) * Pmm) / (i - m);
        Pmm = Pm_plus_1;
        Pm_plus_1 = Plm;
    }
    return Plm;
}
__device__ double* vectorizedCosine(const double* input, size_t size) {

    double* result = new double[size];
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::cos(input[i]);
    }
    return result;
}

__device__ double legendreup(int l, int m, double x, int cl, double Plm, double Plm1m) {
    
    // initialize with Pmm and Pmp1m, then walk upways and pass previous values
    if (cl == abs(m)){
        double Pmm;
        double amm = 1.0;
        for (int k = 1; k<=abs(m); ++k) {
            amm *= (2*k+1)/(2*k);
        }
        amm = power(-1,abs(m))*sqrtf(amm);
        Pmm = amm * sqrtf(power(1 - x * x, abs(m)));
        if (m<0) {
            Pmm = power(-1, abs(m)) * factorial(0) / factorial(2*abs(m)) * Pmm;
        }
        if (cl==l) {
            // we are done
            return Pmm;
        }
        
        double amp1m = 1.0;
        for (int k = 1; k<=abs(m)+1; ++k) {
            amp1m *= (2*k+1)/(2*k);
        }
        amp1m = power(-1,abs(m)) * sqrtf(amp1m);
        double Pmp1m = amp1m * x * sqrtf(power(1 - x * x, abs(m)));
        if (m<0) {
            Pmp1m = power(-1, abs(m)) * factorial(0) / factorial(2*abs(m)) * Pmp1m;
        }

        if (cl == l-1) {
            return Pmp1m;
        } else {
            return legendreup(l, abs(m), x, cl+1, Pmp1m, Pmm);
        }
    }
    // cl starts with m, and goes up to l
    if (cl==l) {
        // we are done
        return Plm;
    } else if (cl > l) {
        // just for safety
        return 0.0;
    } else if (cl == l-1) {
        // we are almost done
        double l_ = l;
        double blm = - sqrtf((2.*l_+1.)/(2.*l_-3)*((l_-1.)*(l_-1.)-m*m)/(l_*l_-m*m));
        double alm = sqrtf((4.*l_*l_-1.)/(l_*l_-m*m));
        double Plp1m = alm * x * Plm + blm * Plm1m;
        return Plp1m;
    } else {
        double blp1m = - sqrtf((2.*(cl+1)+1.)/(2.*(cl+1)-3)*(((cl+1)-1.)*((cl+1)-1.)-m*m)/((cl+1)*(cl+1)-m*m));
        double alp1m = sqrtf((4.*(cl+1)*(cl+1)-1.)/((cl+1)*(cl+1)-m*m));
        double Plp1m = alp1m * x * Plm + blp1m * Plm1m;
        // double Plp1m = alp1m*x*legendreup(l-1, abs(m), x) + blp1m * legendreup(l-2, abs(m), x);
        Plm = Plm1m;
        return legendreup(l, m, x, cl+1, Plp1m, Plm);
        // double Plm = ((cl-m+1)*legendreup(l, m, x, cl+1, 0., 0.) - (cl+m)*legendreup(l, m, x, cl-1, 0., 0.)) / ((2.0*cl+1.0)*x);
    }
}
__device__ double legendreuploop(int l, int m, double x, int cl, double Plm, double Plm1m) {
    // initialize with Pmm and Pmp1m, then walk upways and pass previous values
    if (cl == abs(m)){
        double Pmm;
        double amm = 1.0;
        for (int k = 1; k<=abs(m); ++k) {
            amm *= (2*k+1)/(2*k);
        }
        amm = power(-1,abs(m))*dsqrt(amm);
        Pmm = amm * sqrt(power(1 - x * x, abs(m)));
        if (m<0) {
            Pmm = power(-1, abs(m)) * factorial(0) / factorial(2*abs(m)) * Pmm;
        }
        if (cl==l) {
            // we are done
            return Pmm;
        }
        
        double amp1m = 1.0;
        for (int k = 1; k<=abs(m)+1; ++k) {
            amp1m *= (2*k+1)/(2*k);
        }
        amp1m = power(-1,abs(m)) * sqrtf(amp1m);
        double Pmp1m = amp1m * x * sqrtf(power(1 - x * x, abs(m)));
        if (m<0) {
            Pmp1m = power(-1, abs(m)) * factorial(0) / factorial(2*abs(m)) * Pmp1m;
        }

        if (cl == l-1) {
            return Pmp1m;
        } else {
            return legendreuploop(l, abs(m), x, cl+1, Pmp1m, Pmm);
        }
    }
    // cl starts with m, and goes up to l
    for (cl; cl<=l; ++cl) {
        if (cl==l) {
            // we are done
            return Plm;
        } else if (cl == l-1) {
            // we are almost done
            double l_ = l;
            double blm = - dsqrt((2.*l_+1.)/(2.*l_-3)*((l_-1.)*(l_-1.)-m*m)/(l_*l_-m*m));
            double alm = dsqrt((4.*l_*l_-1.)/(l_*l_-m*m));
            double Plp1m = alm * x * Plm + blm * Plm1m;
            return Plp1m;
        } else {
            double l_ = l+1;
            double blm = - dsqrt((2.*l_+1.)/(2.*l_-3)*((l_-1.)*(l_-1.)-m*m)/(l_*l_-m*m));
            double alm = dsqrt((4.*l_*l_-1.)/(l_*l_-m*m));
            double Plp1m = alm * x * Plm + blm * Plm1m;
            // double Plp1m = alp1m*x*legendreup(l-1, abs(m), x) + blp1m * legendreup(l-2, abs(m), x);
            double buff = Plm;
            Plm = Plp1m;
            Plm1m = buff;
            // return legendreup(l, m, x, cl+1, Plp1m, Plm);
            // double Plm = ((cl-m+1)*legendreup(l, m, x, cl+1, 0., 0.) - (cl+m)*legendreup(l, m, x, cl-1, 0., 0.)) / ((2.0*cl+1.0)*x);
        }
    }
}
__device__ double legendredown_Nath(int l, int m, double x) {
    /*
    always walk towards Pmm (eq. 13 of Nathanael).
    */

    if (l<0 or m>l) {
        return 0.0;
    }
     // continue
    if (l == abs(m)) {
        // we are done
        double Pmm;
        double amm = 1.0;
        for (int k = 1; k<=abs(m); ++k) {
            amm *= (2*k+1)/(2*k);
        }
        amm = power(-1,l)*sqrtf(amm);
        Pmm = sqrt(1/(4*M_PI))* amm * sqrtf(power(1 - x * x, abs(m)));
        if (m<0) {
            Pmm = power(-1, abs(m)) * factorial(l-abs(m)) / factorial(l+abs(m)) * Pmm;
        }
        return Pmm;
    } else if (l > m) {
        // continue
        double blm = - sqrtf((2.*l+1.)/(2.*l-3)*((l-1.)*(l-1.)-m*m)/(l*l-m*m));
        double alm = sqrtf((4.*l*l-1.)/(l*l-m*m));
        double Plm = alm*x*legendredown_Nath(l-1, abs(m), x) + blm * legendredown_Nath(l-2, abs(m), x);
        if (m<0) {
            Plm = power(-1, abs(m)) * factorial(l-abs(m)) / factorial(l+abs(m)) * Plm;
        }
        return Plm;
    }
}
__device__ double aleg(int l, int m, double x) {
    // double norm = sqrt((2.0 * l + 1.0) * factorial(l - m) / (4.0 * M_PI * factorial(l + m)));
    // return legendredown_Nath(l, m, x);
    return legendreuploop(l, m, x, abs(m), 0., 0.);//*norm;
}
__global__ void compute_ALPs(int lmax, int mmax, int size_x, double *x, double *ALPs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size_x) {
        double x_ = x[threadIdx.x];
        ALPs[idx] = aleg(lmax, mmax, x_);
    }
}

extern "C" void associated_legendre(int lmax, int mmax, double *host_x, double *host_result, int size_x) {
    double *device_x, *device_result;

    // const int threadsPerBlock = 256;
    // const int blocksPerGrid = (lmax + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&device_x, size_x * sizeof(double));
    cudaMalloc((void**)&device_result, size_x * sizeof(double));

    cudaMemcpy(device_x, host_x, size_x * sizeof(double), cudaMemcpyHostToDevice);

    // auto start = high_resolution_clock::now();
    compute_ALPs<<<1, size_x>>>(lmax, mmax, size_x, device_x, device_result);
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


__device__ double d_legendredown(int l, double x) {
    if (l == 0) {
        return 1.0f;
    } else if (l == 1) {
        return x;
    } else {
        return ((2.0f * l - 1.0f) * x * d_legendredown(l - 1, x) - (l - 1) * d_legendredown(l - 2, x)) / l;
    }
}
__device__ double d_legendreup(int l, double x, int counter, double Pn, double Pn_minus_1) {
    double Pn_plus_1;

    if (counter == l) {
        if (counter > 1){
            Pn_plus_1 = (2*counter+1)*x/(counter+1) * Pn - counter * Pn_minus_1/(counter+1);
            return Pn_plus_1;
        } else if (counter == 1) {
            return x;
        } else if (counter == 0) {
            return 1.0f;
        }
    } else if (counter < l) {
        if (counter < 2) {
            Pn_minus_1 = 1.0f;
            Pn = x;
            return d_legendreup(l, x, counter+1, Pn, Pn_minus_1);
        } else if (counter >= 2) {
            Pn_plus_1 = (2*counter+1)*x/(counter+1) * Pn - counter * Pn_minus_1/(counter+1);
            return d_legendreup(l, x, counter+1, Pn_plus_1, Pn);
        }
    }
}
__global__ void kernel_legendreup(int n, double *x, int lmax, double *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int counter = 0;
    if (idx < n) {
        double x_ = x[threadIdx.x];
        result[idx] = d_legendreup(blockIdx.x, x_, counter, 1.0, x_);
    }
}
__global__ void kernel_legendredown(int n, double *x, int lmax, double *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double x_ = x[threadIdx.x];
        result[idx] = d_legendredown(blockIdx.x, x_);
    }
}


extern "C" void Legendredown(int lmax, double *host_x, double *host_result, int size_x) {
    double *device_x, *device_result;

    cudaMalloc((void **)&device_x, size_x * sizeof(double));
    cudaMalloc((void **)&device_result, size_x * (lmax + 1) * sizeof(double));

    cudaMemcpy(device_x, host_x, size_x * sizeof(double), cudaMemcpyHostToDevice);

    // int threadsPerBlock = 256;
    // int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    printf(" %d %d\n", lmax, size_x);
    // kernel_legendre<<<blocksPerGrid, threadsPerBlock>>>(size* (lmax + 1), device_x, lmax, device_result);
    auto start = high_resolution_clock::now();
    kernel_legendredown<<<lmax+1, size_x>>>(size_x * (lmax + 1), device_x, lmax, device_result);
    cudaDeviceSynchronize();
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    printf("down, time(mu s): %d\n", (int)duration.count());

    cudaMemcpy(host_result, device_result, size_x * (lmax + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_x);
    cudaFree(device_result);
}

extern "C" void Legendreup(int lmax, double *host_x, double *host_result, int size_x) {
    double *device_x, *device_result;

    cudaMalloc((void **)&device_x, size_x * sizeof(double));
    cudaMalloc((void **)&device_result, size_x * (lmax + 1) * sizeof(double));

    cudaMemcpy(device_x, host_x, size_x * sizeof(double), cudaMemcpyHostToDevice);

    // int threadsPerBlock = 256;
    // int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    printf("%d %d\n", lmax, size_x);

    // cudaEventRecord(start);
    auto start = high_resolution_clock::now();
    kernel_legendreup<<<lmax+1, size_x>>>(size_x * (lmax + 1), device_x, lmax, device_result);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    printf("up, time(mu s): %d\n", (int)duration.count());
    // cudaEventRecord(stop);
    // double milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // cudaDeviceSynchronize();

    cudaMemcpy(host_result, device_result, size_x * (lmax + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(device_x);
    cudaFree(device_result);
}


__global__ void kernel_lambda_lmt(int lmax, int mmax, int size_theta, double *theta,  double *sht) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // sht[idx] = idx;
    if (idx < size_theta) {
        // iterate over all blocks
        if (mmax<0) {
            sht[idx] = dsqrt(2) * normalization_constant(lmax, mmax) * aleg(lmax, mmax, cos(theta[idx]));
        } else if (mmax>0) {
            sht[idx] = dsqrt(2) * normalization_constant(lmax, mmax) * aleg(lmax, mmax, cos(theta[idx]));
        } else {
            sht[idx] = normalization_constant(lmax, mmax) * aleg(lmax, 0, cos(theta[idx]));
        }
    }
}

extern "C" void synthesis_ring(int lmax, int mmax, int Nlat, int Nlon, double *host_result) {
    printf("synthesis_ring\n");
    typedef std::complex<double> Complex;
    double *device_theta;
    double *device_result;
    double interface_result[(2*lmax+1)*Nlat];
    // double *device_coeffs_lm, *device_F;

    int NumBlocks = (Nlat+1024-1)/1024;

    // ordering: tbd
    std::vector<double> coeffs_lm(lmax*(lmax+1)/2);
    std::vector<double> F((2*lmax+1)*Nlat);

    std::vector<double> phi = linspace(0., 2. * M_PI, Nlon);
    std::vector<double> theta = linspace(0., M_PI, Nlat);
    
    auto mesh = meshgrid_1d(phi, theta);

    const auto& Phi = mesh.first;
    const auto& Theta = mesh.second;

    cudaMalloc((void**)&device_theta, Nlat * sizeof(double));
    cudaMalloc((void**)&device_result, Nlat * sizeof(double));
    // std::vector<double> map(Nlon*Nlat, 0.0f);

    // for (int l=0; l<lmax+1;l++){
        // for (int m=-l; m<l+1;m++){


    // Calculate lambda_lmt
    for (size_t ringi=0; ringi<Nlat; ringi++) {
        printf("\tringi: %d\n", ringi);
        int startIndex = ringi * Nlon;
        int endIndex = (ringi + 1) * Nlon;
        // double *thetas = new double[Nlat];
        // for (int i = startIndex; i < endIndex; ++i) {
        //     thetas[i - startIndex] = Theta.data()[i];
        //     printf("%f ", Theta.data()[i]);
        // }
        // printf("\n");
        printf("\tlmax: %d\n", lmax);
        for (size_t m=0; m<=lmax; m++) {
            printf("\t\tm: %d\n", m);
            for (size_t l=(int)sqrt(m*m); l<=lmax; l++) {
                printf("\t\t\tl: %d\n", l);
                cudaMemcpy(device_theta, theta.data(), Nlon * sizeof(double), cudaMemcpyHostToDevice);
                
                kernel_lambda_lmt<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
                cudaDeviceSynchronize();
                cudaMemcpy(interface_result, device_result, Nlat * sizeof(double), cudaMemcpyDeviceToHost);
                printf("\t\t\t");
                for (int i = 0; i < Nlat; ++i) {
                    printf("%.4f ", interface_result[i]);
                }
                printf("\n");
                cudaFree(device_theta);
                cudaFree(device_result);

                //kernel_accumulateF<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
                for (int i = 0; i < Nlat; ++i) {
                    //TODO sum over all ls properly
                    F[i] += interface_result[i] * coeffs_lm[i];
                }
            }
        }
    }
    //kernel_FT<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
    for (int i = 0; i < Nlat; ++i) {
        //TODO this is "FFT"
        host_result[i] += interface_result[i]*dsqrt(2) * sin(abs(mmax) * phi_);
    }
}

__global__ void kernel_sht_NlonNlat(int lmax, int mmax, int size_phi, int size_theta, double *phi, double *theta,  double *sht) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    if (idx < size_phi*size_theta) {
        // iterate over all blocks
        double phi_ = phi[idx];
        if (mmax<0) {
            sht[idx] = dsqrt(2) * normalization_constant(lmax, mmax) * aleg(lmax, mmax, cos(theta[idx])) * sin(abs(mmax) * phi_);
        } else if (mmax>0) {
            sht[idx] = dsqrt(2) * normalization_constant(lmax, mmax) * aleg(lmax, mmax, cos(theta[idx])) * cos(mmax * phi_);
        } else {
            sht[idx] = normalization_constant(lmax, mmax) * aleg(lmax, 0, cos(theta[idx]));
        }
    }
}

extern "C" void synthesis_NlonNlat(int lmax, int mmax, int Nlat, int Nlon, double *host_result) {
    typedef std::complex<double> Complex;
    double *device_phi, *device_theta;
    double *device_result;

    std::vector<double> phi = linspace(0., 2. * M_PI, Nlon);
    std::vector<double> theta = linspace(0., M_PI, Nlat);
    
    auto mesh = meshgrid_1d(phi, theta);

    const auto& Phi = mesh.first;
    const auto& Theta = mesh.second;

    // printf("phi: ");
    // for (size_t i = 0; i < Phi.size(); ++i) {
    //     printf("%.3f ", Phi.data()[i]);
    // }
    // printf("\n--\n");
    // printf("theta: ");
    // for (size_t i = 0; i < Theta.size(); ++i) {
    //     printf("%.3f ", Theta.data()[i]);
    // }
    // printf("\n--\n");

    cudaMalloc((void**)&device_phi, Nlon*Nlat * sizeof(double));
    cudaMalloc((void**)&device_theta, Nlon*Nlat * sizeof(double));
    cudaMalloc((void**)&device_result, Nlon*Nlat * sizeof(double));
    // std::vector<double> map(Nlon*Nlat, 0.0f);

    // for (int l=0; l<lmax+1;l++){
        // for (int m=-l; m<l+1;m++){

    cudaMemcpy(device_phi, Phi.data(), Nlon*Nlat * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_theta, Theta.data(), Nlon*Nlat * sizeof(double), cudaMemcpyHostToDevice);
    // kernel_sht<<<Nlat, Nlon>>>(lmax, mmax, Nlon, Nlat, device_phi, device_theta, device_result);
    kernel_sht_NlonNlat<<<Nlat, Nlon>>>(lmax, mmax, Nlon, Nlat, device_phi, device_theta, device_result);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result, device_result, Nlon*Nlat * sizeof(double), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < Nlon*Nlat; ++i) {
    //     printf("%.4f ", host_result[i]);
    //     }
    // printf("\n-----\n");
    cudaFree(device_phi);
    cudaFree(device_result);
    // printf("\nPhi: ");
    // for (int i = 0; i < Nlon; ++i) {
    //     for (int j = 0; j < Nlat; ++j) {
    //         printf("%f ", Phi[i][j]);
    //     }
    // }
}

// extern "C" void anslysis(int lmax, int mmax, double *host_x, double *host_result, int size_x) {
//     double *device_x, *device_result;
//     int Nlat = 1000;
//     int Nlon = 1000;
//     double theta = linspace(0, np.pi, Nlat)
//     double phi = linspace(0, 2 * np.pi, Nlon)
//     auto mesh = meshgrid(phi, theta);
//     phi = mesh.first;
//     theta = mesh.second;
//     // const int threadsPerBlock = 256;
//     // const int blocksPerGrid = (lmax + threadsPerBlock - 1) / threadsPerBlock;

//     cudaMalloc((void**)&device_x, size_x * sizeof(double));
//     cudaMalloc((void**)&device_result, size_x * sizeof(double));
//     std::vector<std::vector<Complex>> complexArray(rows, std::vector<Complex>(cols));
//     std::vector<std::vector<Complex>> coeffs(Nlat, std::vector<Complex>(Nlon));
//     cudaMemcpy(device_x, host_x, size_x * sizeof(double), cudaMemcpyHostToDevice);

//     // auto start = high_resolution_clock::now();
//     for l in range(lmax + 1):
//     for m in range(-l, l + 1):
//         coeff_sum = 0.0
//         for ringi in range(Nlat):
//             for j in range(Nlon):
//                 theta_val = theta[ringi, j]
//                 std::vector<int> slicedColumn;
//                 phi_val = phi[ringi, j]
//                 kernel_sht<<<1, size_x>>>(lmax, mmax, size_x, device_x, device_result);
//                 cudaDeviceSynchronize();
//                 cudaMemcpy(host_result, device_result, size_x * sizeof(double), cudaMemcpyDeviceToHost);
                
//                 coeff_sum += grid_data[i, j] * np.conj(host_result)

//     cudaFree(device_x);
//     cudaFree(device_result);
//     cudaDeviceSynchronize();
//     // auto stop = high_resolution_clock::now();

//     cudaMemcpy(host_result, device_result, size_x * sizeof(double), cudaMemcpyDeviceToHost);

//     cudaFree(device_x);
//     cudaFree(device_result);
// }


        // std::vector<double> sp;
        // for (size_t i = 0; i < Phi.size(); ++i) {
        //     sp.push_back(Phi[i][ringi]);
        // }
        // std::vector<double> st;
        // for (size_t i = 0; i < Theta.size(); ++i) {
        //     st.push_back(Theta[i][ringi]);
        // }


    // for (int ringi=0; ringi<Nlat; ringi++){
    //     printf("phi: ");
    //     for (size_t i = 0; i < Phi.size(); ++i) {
    //         printf("%.3f ", Phi.data()[i]);
    //     }
    //     printf("\n");
    //     printf("theta: ");
    //     for (size_t i = 0; i < Theta.size(); ++i) {
    //         printf("%.3f ", Theta.data()[i]);
    //     }
    //     printf("\n");
    // }