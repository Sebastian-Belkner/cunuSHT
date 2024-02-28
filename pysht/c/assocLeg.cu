#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>
#include <complex>
#include <random>
#include <cuda_runtime.h>

using namespace std::chrono;
typedef std::complex<double> Complex;

int habs(int val) {
    return abs(val);
}
double hpower(double base, int exponent) {
    double result = 1.0;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}
std::vector<Complex> random_alm(int size) {
    // Create a random number generator for the real and imaginary parts
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the generator
    std::uniform_real_distribution<> real_dist(-1.0, 1.0); // 
    std::uniform_real_distribution<> imag_dist(-1.0, 1.0); // 

    std::vector<Complex> randomComplexArray;
    // Generate random complex numbers and store them in the vector
    for (int i = 0; i < size; ++i) {
        double realPart = real_dist(eng); // Generate a random real part
        double imagPart = imag_dist(eng); // Generate a random imaginary part
        Complex randomNumber(realPart, imagPart); // Create a complex number
        randomComplexArray.push_back(randomNumber); // Store the complex number in the vector
    }
    return randomComplexArray;
}
double hfactorial(int n) {
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}
double hnormalization_constant(int l, int m) {
    return sqrt((2 * l + 1) * hfactorial(l - m) / (4 * M_PI * hfactorial(l + m)));
}
int getlmax(int size) {
    int x = (-3 + sqrt(1 + 8 * size)) / 2;
    return (int)x;
}
int getidx(int lmax, int l, int m) {
    int idx = 0;
    idx = m * (2 * lmax + 1 - m) / 2 + l;
    return (int)idx;
}
int getsize(int lmax){
    int mmax = lmax;
    int size = mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1;
    return (int)size;
}

std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}
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

__device__ double dsqrt(double n) {
    return sqrt(n);
}
__device__ int dabs(double n) {
    return abs(n);
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


__device__ double legendreuploop(int l, int m, double x, int cl, double Plm, double Plm1m) {
    // initialize with Pmm and Pmp1m, then walk upways and pass previous values
    if (cl == abs(m)){
        double Pmm;
        double amm = 1.0;
        for (int k = 1; k<=dabs(m); ++k) {
            amm *= (2*k+1)/(2*k);
        }
        amm = power(-1,dabs(m))*dsqrt(amm);
        Pmm = amm * sqrt(power(1 - x * x, abs(m)));
        if (m<0) {
            Pmm = power(-1, dabs(m)) * factorial(0) / factorial(2*dabs(m)) * Pmm;
        }
        if (cl==l) {
            // we are done
            return Pmm;
        }
        
        double amp1m = 1.0;
        for (int k = 1; k<=dabs(m)+1; ++k) {
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
            return legendreuploop(l, dabs(m), x, cl+1, Pmp1m, Pmm);
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
            double buff = Plm;
            Plm = Plp1m;
            Plm1m = buff;
        }
    }
}
__device__ double aleg(int l, int m, double x) {
    // return legendredown_Nath(l, m, x);
    return legendreuploop(l, m, x, abs(m), 0., 0.);
}
__global__ void compute_ALPs(int lmax, int mmax, int size_x, double *x, double *ALPs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size_x) {
        double x_ = x[threadIdx.x];
        ALPs[idx] = aleg(lmax, mmax, x_);
    }
}

__device__ double* legendreuplooptrace(int l, int m, double x, double *result) {
    // initialize with Pmm and Pmp1m, then walk upways and pass previous values, store result 
    // double Plm, Plm1m;
    int cl = dabs(m)+1;

    // cl starts with m, and goes up to l
    for (; cl<=l; cl++) {
        double l_ = l;
        double blm = - dsqrt((2.*l_+1.)/(2.*l_-3)*((l_-1.)*(l_-1.)-m*m)/(l_*l_-m*m));
        double alm = dsqrt((4.*l_*l_-1.)/(l_*l_-m*m));
        double Plp1m = alm * x * result[cl-m+1-1] + blm * result[cl-m+1-2];
        result[cl-m+1] = Plp1m;
    }
    return result;
}
__device__ double* alegtrace(const int l, const int m, double x) {
    // return legendredown_Nath(l, m, x);
    double *result = new double[l-dabs(m)+1];
    double Pmm;

    double amm = 1.0;
    for (int k = 1; k<=dabs(m); ++k) {
        amm *= (2*k+1)/(2*k);
    }
    amm = power(-1,dabs(m))*dsqrt(amm);
    Pmm = amm * sqrt(power(1 - x * x, dabs(m)));
    if (m<0) {
        Pmm = power(-1, dabs(m)) * factorial(0) / factorial(2*dabs(m)) * Pmm;
    }
    result[0] = Pmm;
    if (m==l) {
        // we are done
        return result;
    }
    
    double amp1m = 1.0;
    for (int k = 1; k<=dabs(m)+1; ++k) {
        amp1m *= (2*k+1)/(2*k);
    }
    amp1m = power(-1,dabs(l)) * sqrtf(amp1m);
    double Pmp1m = amp1m * x * sqrtf(power(1 - x * x, dabs(m)));
    if (m<0) {
        Pmp1m = power(-1, dabs(m)) * factorial(0) / factorial(2*dabs(m)) * Pmp1m;
    }
    result[1] = Pmp1m;
    if (m == l-1) {
        return result;
    } else {
        return legendreuplooptrace(l, m, x, result);
    }
}
__global__ void kernel_lambda_lmt_tl(const int lmax, const int mmax, int size_theta, double *theta, double *sht) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= size_theta*(lmax-abs(mmax)+1)) {
        double *buff = alegtrace(lmax, mmax, cos(theta[idx]));
        for (size_t i=0; i<=lmax-abs(mmax); i++) {
            sht[idx+i*size_theta] = buff[i]; //dsqrt(2) * normalization_constant(abs(mmax)+i, abs(mmax)) * 
            __syncthreads();
        }
    }
}
extern "C" void synthesis_ringl(int size_alm, int Nlat, int Nlon, double *host_result, int l_, int m_) {
    printf("synthesis_ring\n");
    double *device_theta, *device_result;
    
    // std::vector<Complex> alm = random_alm(size_alm);
    // int lmax = getlmax(size_alm); //sizeof(alm)/sizeof(alm[0])
    // for (int i = 0; i < size_alm; ++i) {
    //     printf("%.2f+%.2fi, ", alm[i].real(), alm[i].imag());
    // }
    // printf("\n");

    std::vector<Complex> alm(size_alm);
    int lmax = getlmax(size_alm); //sizeof(alm)/sizeof(alm[0])
    for (int i = 0; i < size_alm; ++i) {
        alm[i] = Complex(0.0, 0.0);
    }
    for (size_t l=0; l<=lmax;l++){
        if (l==l_){
            alm[getidx(lmax,l,m_)] = Complex(1.0, 1.0);
        }
    }

    for (int i = 0; i < size_alm; ++i) {
        printf("%.2f+%.2fi, ", alm[i].real(), alm[i].imag());
    }
    printf("\n");
    
    int NumBlocks = (Nlat+1024-1)/1024;
    
    std::vector<double> phi = linspace(0., 2. * M_PI, Nlon);
    std::vector<double> theta = linspace(0., M_PI, Nlat);
    auto mesh = meshgrid_1d(phi, theta);
    const auto& Phi = mesh.first;
    const auto& Theta = mesh.second;
    int size_f = (lmax+1)*Nlat;
    std::vector<std::complex<double>> F(size_f, std::complex<double>(0.0, 0.0));

    cudaMalloc((void**)&device_theta, Nlat * sizeof(double));
    cudaMemcpy(device_theta, theta.data(), Nlat * sizeof(double), cudaMemcpyHostToDevice);
    
    for (size_t m=0; m<=lmax; m++) {
        int m_max = m==0?1:2;
        for (size_t m_=0; m_<m_max; m_++){
            double interface_result[(lmax-m+1) * Nlat];
            cudaMalloc((void**)&device_result, (lmax-m+1) * Nlat * sizeof(double));
            int mmax_ = m_%2==0?1*m:-1*m;
            kernel_lambda_lmt_tl<<<NumBlocks, 1024>>>(lmax, mmax_, Nlat, device_theta, device_result);
            cudaDeviceSynchronize();
            cudaMemcpy(interface_result, device_result, (lmax-m+1) * Nlat * sizeof(double), cudaMemcpyDeviceToHost);
            
            // printf("(lmax, mmax), lambda_lmt(%d, %d)= ", lmax, mmax_);
            // for (int i = 0; i < (lmax-habs(m)+1) * Nlat; i++) {
            //     printf("%.2f, ", interface_result[i]);
            // }
            // printf("\n");

            //kernel_accumulateF<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
            for (size_t l=m; l<=lmax; l++) {
                // printf("l=%d, m=%d, getidx()=%d\n", l,m,getidx(lmax, l, m));
                for (size_t ringi = 0; ringi < Nlat; ++ringi) {
                    // printf("%d ", l*Nlat+ringi);
                    if (m_==0) {
                        F[m*Nlat+ringi] += alm[getidx(lmax, l, m)] * interface_result[(l-m)*Nlat+ringi];
                    } else {
                        F[m*Nlat+ringi] += conj(alm[getidx(lmax, l, m)]) * interface_result[(l-m)*Nlat+ringi];
                    }
                }
                // printf("\n");
                // printf("+F(%d)= ", m_%2==0?1*m:-1*m);
                // for (int i = 0; i < (lmax+1)*Nlat; i++) {
                //     printf("%.2f, ", F[i]);
                // }printf("\n\n");
            }
        }
    }
    cudaFree(device_theta);
    cudaFree(device_result);
    //kernel_FT<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
    for (size_t m=0; m<=lmax; m++) {
        for (size_t ringi=0; ringi<Nlat; ringi++) {
            for (size_t x=0;x<Nlon; x++) {
                double phi_ = Phi[x];
                host_result[ringi*Nlat+x] += real(F[m*Nlat+ringi])*cos(m*phi_);
            }
        }
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

    // Free memory
    cudaFree(device_x);
    cudaFree(device_result);
}