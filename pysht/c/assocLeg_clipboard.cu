#include <stdio.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <cmath>
#include <memory>
#include <complex>
#include <random>

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

__device__ *double legendreuplooptrace(int l, int m, double x, int cl, double Plm, double Plm1m, double *result) {
    // initialize with Pmm and Pmp1m, then walk upways and pass previous values, store result 
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
        result[l-cl] = Pmm;
        if (cl==l) {
            // we are done
            return result;
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
        result[l-cl+1] = Pmp1m;
        if (cl == l-1) {
            return result;
        } else {
            return legendreuploop(l, abs(m), x, cl+1, Pmp1m, Pmm, result);
        }
    }
    // cl starts with m, and goes up to l
    for (cl; cl<=l; ++cl) {
        if (cl==l) {
            // we are done
            return result;
        } else if (cl == l-1) {
            // we are almost done
            double l_ = l;
            double blm = - dsqrt((2.*l_+1.)/(2.*l_-3)*((l_-1.)*(l_-1.)-m*m)/(l_*l_-m*m));
            double alm = dsqrt((4.*l_*l_-1.)/(l_*l_-m*m));
            double Plp1m = alm * x * Plm + blm * Plm1m;
            result[cl-l+1] = Plp1m;
            return result;
        } else {
            double l_ = l+1;
            double blm = - dsqrt((2.*l_+1.)/(2.*l_-3)*((l_-1.)*(l_-1.)-m*m)/(l_*l_-m*m));
            double alm = dsqrt((4.*l_*l_-1.)/(l_*l_-m*m));
            double Plp1m = alm * x * Plm + blm * Plm1m;
            double buff = Plm;
            Plm = Plp1m;
            Plm1m = buff;
            result[cl-l+1] = Plp1m;
        }
    }
}
__device__ *double alegtrace(int l, int m, double x) {
    // return legendredown_Nath(l, m, x);
    return legendreuplooptrace(l, m, x, abs(m), 0., 0.);
}
__global__ void kernel_lambda_lmt_tl(int lmax, int mmax, int size_theta, double *theta,  double *sht) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_theta*(lmax-mmax)) {
        // iterate over all blocks
        double *buff = dsqrt(2) * normalization_constant(lmax, mmax) * alegtrace(lmax, mmax, cos(theta[idx]));
        for (size_t i; i<lmax-mmax; i++) {
            sht[idx*lmax-mmax+i] = buff[i];
        }
    }
}
extern "C" void synthesis_ringl(int size_alm, int Nlat, int Nlon, double *host_result) {
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
    int m_ = 1;
    for (size_t l=0; l<=lmax;l++){
        // if (l>=2*m_){
            alm[getidx(lmax,l,m_)] = Complex(1.0, 1.0);
        // }
    }

    for (int i = 0; i < size_alm; ++i) {
        printf("%.2f+%.2fi, ", alm[i].real(), alm[i].imag());
    }
    printf("\n");
    double interface_result[(2*lmax+1)*Nlat];
    int NumBlocks = (Nlat+1024-1)/1024;
    
    std::vector<double> phi = linspace(0., 2. * M_PI, Nlon);
    std::vector<double> theta = linspace(0., M_PI, Nlat);
    auto mesh = meshgrid_1d(phi, theta);
    const auto& Phi = mesh.first;
    const auto& Theta = mesh.second;
    int size_f = (lmax+1)*Nlat;
    std::vector<std::complex<double>> F(size_f, std::complex<double>(0.0, 0.0));
    
    for (size_t m=0; m<=lmax; m++) {
        for (size_t m_=0; m<2; m++){
            // for (size_t l=m; l<=lmax; l++) {
            cudaMalloc((void**)&device_theta, Nlat * sizeof(double));
            cudaMalloc((void**)&device_result, (lmax-m) * Nlat * sizeof(double));
            cudaMemcpy(device_theta, theta.data(), Nlat * sizeof(double), cudaMemcpyHostToDevice);
            kernel_lambda_lmt_tl<<<NumBlocks, 1024>>>(lmax, hpower(-1,m_)*m_, Nlat, device_theta, device_result);
            cudaDeviceSynchronize();
            cudaMemcpy(interface_result, device_result, (lmax-m) * Nlat * sizeof(double), cudaMemcpyDeviceToHost);

            //kernel_accumulateF<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
            for (size_t l=m; l<=lmax; l++) {
                for (int ringi = 0; ringi < Nlat; ++ringi) {
                    if (m_==0) {
                        F[m*Nlat+ringi] += alm[getidx(lmax, l, m)] * interface_result[m*Nlat+ringi];
                    } else {
                        F[m*Nlat+ringi] += conj(alm[getidx(lmax, l, m)]) * interface_result[m*Nlat+ringi];
                    }
                }
                cudaFree(device_theta);
                cudaFree(device_result);
            }
        }
    }
    //kernel_FT<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
    for (size_t m=0; m<=lmax; m++) {
        for (size_t ringi=0; ringi<Nlat; ringi++) {
            for (size_t y=0;y<Nlon; y++) {
                double phi_ = Phi[y];
                host_result[ringi*Nlat+y] += real(F[m*Nlat+ringi])*cos(m*phi_);
            }
        }
    }
}

__global__ void kernel_lambda_lmt_allt(int lmax, int mmax, int size_theta, double *theta,  double *sht) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size_theta) {
        // iterate over all blocks
        sht[idx] = dsqrt(2) * normalization_constant(lmax, mmax) * aleg(lmax, mmax, cos(theta[idx]));
        // sht[idx] += dsqrt(2) * normalization_constant(lmax, mmax) * aleg(lmax, -mmax, cos(theta[idx]));
    }
}
extern "C" void synthesis_ring(int size_alm, int Nlat, int Nlon, double *host_result) {
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
    int m_ = 1;
    for (size_t l=0; l<=lmax;l++){
        // if (l>=2*m_){
            alm[getidx(lmax,l,m_)] = Complex(1.0, 1.0);
        // }
    }

    for (int i = 0; i < size_alm; ++i) {
        printf("%.2f+%.2fi, ", alm[i].real(), alm[i].imag());
    }
    printf("\n");
    double interface_result[(2*lmax+1)*Nlat];
    int NumBlocks = (Nlat+1024-1)/1024;
    
    std::vector<double> phi = linspace(0., 2. * M_PI, Nlon);
    std::vector<double> theta = linspace(0., M_PI, Nlat);
    auto mesh = meshgrid_1d(phi, theta);
    const auto& Phi = mesh.first;
    const auto& Theta = mesh.second;
    int size_f = (lmax+1)*Nlat;
    std::vector<std::complex<double>> F(size_f, std::complex<double>(0.0, 0.0));
    
    for (size_t m=0; m<=lmax; m++) {
        for (size_t m_=0; m<2; m++){
            for (size_t l=m; l<=lmax; l++) {
                cudaMalloc((void**)&device_theta, Nlat * sizeof(double));
                cudaMalloc((void**)&device_result, Nlat * sizeof(double));
                cudaMemcpy(device_theta, theta.data(), Nlat * sizeof(double), cudaMemcpyHostToDevice);
                kernel_lambda_lmt_allt<<<NumBlocks, 1024>>>(l, hpower(-1,m_)*m_, Nlat, device_theta, device_result);
                cudaDeviceSynchronize();
                cudaMemcpy(interface_result, device_result, Nlat * sizeof(double), cudaMemcpyDeviceToHost);
                //kernel_accumulateF<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
                for (int ringi = 0; ringi < Nlat; ++ringi) {
                    if (m_==0) {
                        F[m*Nlat+ringi] += alm[getidx(lmax, l, m)] * interface_result[ringi];
                    } else {
                        F[m*Nlat+ringi] += conj(alm[getidx(lmax, l, m)]) * interface_result[ringi];
                    }
                }
                cudaFree(device_theta);
                cudaFree(device_result);
            }
        }
    }
    //kernel_FT<<<NumBlocks, 1024>>>(lmax, mmax, Nlat, device_theta, device_result);
    for (size_t m=0; m<=lmax; m++) {
        for (size_t ringi=0; ringi<Nlat; ringi++) {
            for (size_t y=0;y<Nlon; y++) {
                double phi_ = Phi[y];
                host_result[ringi*Nlat+y] += real(F[m*Nlat+ringi])*cos(m*phi_);
            }
        }
    }
}

__global__ void kernel_sht_NlonNlat(int lmax, int mmax, int size_phi, int size_theta, double *phi, double *theta,  double *sht) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
