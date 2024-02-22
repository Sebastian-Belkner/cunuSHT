#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>
#include <complex>

using namespace std::chrono;


// float* linspace(float start, float stop, int num_points) {
//     float* result = new float[num_points];
//     float step = (stop - start) / (num_points - 1);
//     for (int i = 0; i < num_points; ++i) {
//         result[i] = start + i * step;
//     }
//     return result;
// }

std::vector<float> linspace(float start, float end, int num) {
    std::vector<float> result(num);
    float step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// std::vector<float> vectorizedCosine(const std::vector<float>& input) {
//     std::vector<float> result;
//     result.reserve(input.size());

//     for (float val : input) {
//         result.push_back(std::cos(val));
//     }

//     return result;
// }


std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> meshgrid(const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<std::vector<float>> X(x.size(), std::vector<float>(y.size()));
    std::vector<std::vector<float>> Y(x.size(), std::vector<float>(y.size()));

    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < y.size(); ++j) {
            X[i][j] = x[i];
            Y[i][j] = y[j];
        }
    }
    return std::make_pair(X, Y);
}

std::pair<std::vector<float>, std::vector<float>> meshgrid_1d(const std::vector<float>& x, const std::vector<float>& y) {
    std::vector<float>X(x.size()*y.size());
    std::vector<float>Y(x.size()*y.size());

    for (size_t i = 0; i < y.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            X[i*x.size()+j] = x[j];
            Y[i*x.size()+j] = y[i];
        }
    }
    return std::make_pair(X, Y);
}

__device__ float factorial(int n) {
    float result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}
__device__ float power(float base, int exponent) {
    float result = 1.0;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}
__device__ float normalization_constant(int l, int m) {
    return sqrtf((2 * l + 1) * factorial(l - m) / (4 * M_PI * factorial(l + m)));
}
__device__ float legendre(int l, int m, float x) {
    float fact = factorial(l - m);
    float Pmm = power(-1, m) * fact * power(1.0 - x * x, m / 2.0);
    float Pm_minus_1 = power(-1, m) * (2.0 * m - 1.0) * power(1.0 - x * x, m / 2.0 - 1.0);
    if (l == m) {
        return Pmm;
    }
    float Pm_plus_1 = power(-1, m) * (2.0 * m + 1.0) * x * Pm_minus_1;
    if (l == m + 1) {
        return Pm_plus_1;
    }
    float Plm;
    for (int i = m + 2; i <= l; ++i) {
        Plm = (x * (2.0 * i - 1.0) * Pm_plus_1 - (i + m - 1.0) * Pmm) / (i - m);
        Pmm = Pm_plus_1;
        Pm_plus_1 = Plm;
    }
    return Plm;
}
__device__ float* vectorizedCosine(const float* input, size_t size) {

    float* result = new float[size];
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::cos(input[i]);
    }
    return result;
}

__device__ float legendreup(int l, int m, float x, int cl, float Plm, float Plm1m) {
    
    // initialize with Pmm and Pmp1m, then walk upways and pass previous values
    if (cl == abs(m)){
        float Pmm;
        float amm = 1.0;
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
        
        float amp1m = 1.0;
        for (int k = 1; k<=abs(m)+1; ++k) {
            amp1m *= (2*k+1)/(2*k);
        }
        amp1m = power(-1,abs(m)) * sqrtf(amp1m);
        float Pmp1m = amp1m * x * sqrtf(power(1 - x * x, abs(m)));
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
        float l_ = l;
        float blm = - sqrtf((2.*l_+1.)/(2.*l_-3)*((l_-1.)*(l_-1.)-m*m)/(l_*l_-m*m));
        float alm = sqrtf((4.*l_*l_-1.)/(l_*l_-m*m));
        float Plp1m = alm * x * Plm + blm * Plm1m;
        return Plp1m;
    } else {
        float blp1m = - sqrtf((2.*(cl+1)+1.)/(2.*(cl+1)-3)*(((cl+1)-1.)*((cl+1)-1.)-m*m)/((cl+1)*(cl+1)-m*m));
        float alp1m = sqrtf((4.*(cl+1)*(cl+1)-1.)/((cl+1)*(cl+1)-m*m));
        float Plp1m = alp1m * x * Plm + blp1m * Plm1m;
        // float Plp1m = alp1m*x*legendreup(l-1, abs(m), x) + blp1m * legendreup(l-2, abs(m), x);
        Plm = Plm1m;
        return legendreup(l, m, x, cl+1, Plp1m, Plm);
        // float Plm = ((cl-m+1)*legendreup(l, m, x, cl+1, 0., 0.) - (cl+m)*legendreup(l, m, x, cl-1, 0., 0.)) / ((2.0*cl+1.0)*x);
    }
}
__device__ float legendredown_Nath(int l, int m, float x) {
    /*
    always walk towards Pmm (eq. 13 of Nathanael).
    */

    if (l<0 or m>l) {
        return 0.0;
    }
     // continue
    if (l == abs(m)) {
        // we are done
        float Pmm;
        float amm = 1.0;
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
        float blm = - sqrtf((2.*l+1.)/(2.*l-3)*((l-1.)*(l-1.)-m*m)/(l*l-m*m));
        float alm = sqrtf((4.*l*l-1.)/(l*l-m*m));
        float Plm = alm*x*legendredown_Nath(l-1, abs(m), x) + blm * legendredown_Nath(l-2, abs(m), x);
        if (m<0) {
            Plm = power(-1, abs(m)) * factorial(l-abs(m)) / factorial(l+abs(m)) * Plm;
        }
        return Plm;
    }
}
__device__ float aleg(int l, int m, float x) {
    // float norm = sqrt((2.0 * l + 1.0) * factorial(l - m) / (4.0 * M_PI * factorial(l + m)));
    // return legendredown_Nath(l, m, x);
    return legendreup(l, m, x, abs(m), 0., 0.);//*norm;
}
__global__ void compute_ALPs(int lmax, int mmax, int size_x, float *x, float *ALPs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size_x) {
        float x_ = x[threadIdx.x];
        ALPs[idx] = aleg(lmax, mmax, x_);
    }
}

extern "C" void associated_legendre(int lmax, int mmax, float *host_x, float *host_result, int size_x) {
    float *device_x, *device_result;

    // const int threadsPerBlock = 256;
    // const int blocksPerGrid = (lmax + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&device_x, size_x * sizeof(float));
    cudaMalloc((void**)&device_result, size_x * sizeof(float));

    cudaMemcpy(device_x, host_x, size_x * sizeof(float), cudaMemcpyHostToDevice);

    // auto start = high_resolution_clock::now();
    compute_ALPs<<<1, size_x>>>(lmax, mmax, size_x, device_x, device_result);
    cudaDeviceSynchronize();
    // auto stop = high_resolution_clock::now();

    cudaMemcpy(host_result, device_result, size_x * sizeof(float), cudaMemcpyDeviceToHost);

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

    // Free memory
    cudaFree(device_x);
    cudaFree(device_result);
}


__device__ float d_legendredown(int l, float x) {
    if (l == 0) {
        return 1.0f;
    } else if (l == 1) {
        return x;
    } else {
        return ((2.0f * l - 1.0f) * x * d_legendredown(l - 1, x) - (l - 1) * d_legendredown(l - 2, x)) / l;
    }
}
__device__ float d_legendreup(int l, float x, int counter, float Pn, float Pn_minus_1) {
    float Pn_plus_1;

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
__global__ void kernel_legendreup(int n, float *x, int lmax, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int counter = 0;
    if (idx < n) {
        float x_ = x[threadIdx.x];
        result[idx] = d_legendreup(blockIdx.x, x_, counter, 1.0, x_);
    }
}
__global__ void kernel_legendredown(int n, float *x, int lmax, float *result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x_ = x[threadIdx.x];
        result[idx] = d_legendredown(blockIdx.x, x_);
    }
}


extern "C" void Legendredown(int lmax, float *host_x, float *host_result, int size_x) {
    float *device_x, *device_result;

    cudaMalloc((void **)&device_x, size_x * sizeof(float));
    cudaMalloc((void **)&device_result, size_x * (lmax + 1) * sizeof(float));

    cudaMemcpy(device_x, host_x, size_x * sizeof(float), cudaMemcpyHostToDevice);

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

    cudaMemcpy(host_result, device_result, size_x * (lmax + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_x);
    cudaFree(device_result);
}

extern "C" void Legendreup(int lmax, float *host_x, float *host_result, int size_x) {
    float *device_x, *device_result;

    cudaMalloc((void **)&device_x, size_x * sizeof(float));
    cudaMalloc((void **)&device_result, size_x * (lmax + 1) * sizeof(float));

    cudaMemcpy(device_x, host_x, size_x * sizeof(float), cudaMemcpyHostToDevice);

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
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // cudaDeviceSynchronize();

    cudaMemcpy(host_result, device_result, size_x * (lmax + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_x);
    cudaFree(device_result);
}


__global__ void kernel_sht(int lmax, int mmax, int size_phi, int size_theta, float *phi, float *theta,  float *sht) {
    // TODO phi is x_ but wrong unit
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x; 
    if (idx < size_phi*size_theta) {
        // iterate over all blocks
        float phi_ = phi[idx];
        // float *cost = vectorizedCosine(theta, size_theta);
        if (mmax<0) {
            // std::vector<float> cosmp = vectorizedCosine(mmax * phi_);
            sht[idx] = sqrtf(2) * normalization_constant(lmax, mmax) * aleg(lmax, mmax, cosf(theta[idx])) * sinf(abs(mmax) * phi_);
        } else if (mmax>0) {
            sht[idx] = sqrtf(2) * normalization_constant(lmax, mmax) * aleg(lmax, mmax, cosf(theta[idx])) * cosf(mmax * phi_);
        } else {
            sht[idx] = normalization_constant(lmax, mmax) * aleg(lmax, 0, cosf(theta[idx]));

        }
    }
}


extern "C" void synthesis(int lmax, int mmax, int Nlat, int Nlon, float *host_result) {
    typedef std::complex<float> Complex;
    float *device_phi, *device_theta;
    float *device_result;

    std::vector<float> phi = linspace(0., 2. * M_PI, Nlon);
    std::vector<float> theta = linspace(0., M_PI, Nlat);
    
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

    cudaMalloc((void**)&device_phi, Nlon*Nlat * sizeof(float));
    cudaMalloc((void**)&device_theta, Nlon*Nlat * sizeof(float));
    cudaMalloc((void**)&device_result, Nlon*Nlat * sizeof(float));
    // std::vector<float> map(Nlon*Nlat, 0.0f);

    // for (int l=0; l<lmax+1;l++){
        // for (int m=-l; m<l+1;m++){

    cudaMemcpy(device_phi, Phi.data(), Nlon*Nlat * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_theta, Theta.data(), Nlon*Nlat * sizeof(float), cudaMemcpyHostToDevice);
    kernel_sht<<<Nlat, Nlon>>>(lmax, mmax, Nlon, Nlat, device_phi, device_theta, device_result);
    cudaDeviceSynchronize();
    cudaMemcpy(host_result, device_result, Nlon*Nlat * sizeof(float), cudaMemcpyDeviceToHost);
    
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

// extern "C" void anslysis(int lmax, int mmax, float *host_x, float *host_result, int size_x) {
//     float *device_x, *device_result;
//     int Nlat = 1000;
//     int Nlon = 1000;
//     float theta = linspace(0, np.pi, Nlat)
//     float phi = linspace(0, 2 * np.pi, Nlon)
//     auto mesh = meshgrid(phi, theta);
//     phi = mesh.first;
//     theta = mesh.second;
//     // const int threadsPerBlock = 256;
//     // const int blocksPerGrid = (lmax + threadsPerBlock - 1) / threadsPerBlock;

//     cudaMalloc((void**)&device_x, size_x * sizeof(float));
//     cudaMalloc((void**)&device_result, size_x * sizeof(float));
//     std::vector<std::vector<Complex>> complexArray(rows, std::vector<Complex>(cols));
//     std::vector<std::vector<Complex>> coeffs(Nlat, std::vector<Complex>(Nlon));
//     cudaMemcpy(device_x, host_x, size_x * sizeof(float), cudaMemcpyHostToDevice);

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
//                 cudaMemcpy(host_result, device_result, size_x * sizeof(float), cudaMemcpyDeviceToHost);
                
//                 coeff_sum += grid_data[i, j] * np.conj(host_result)

//     cudaFree(device_x);
//     cudaFree(device_result);
//     cudaDeviceSynchronize();
//     // auto stop = high_resolution_clock::now();

//     cudaMemcpy(host_result, device_result, size_x * sizeof(float), cudaMemcpyDeviceToHost);

//     cudaFree(device_x);
//     cudaFree(device_result);
// }


        // std::vector<float> sp;
        // for (size_t i = 0; i < Phi.size(); ++i) {
        //     sp.push_back(Phi[i][ringi]);
        // }
        // std::vector<float> st;
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