#include <stddef.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename Scalar>
__global__ void compute_doubling_spin0_2D(Scalar* synth2D, const size_t nring, const size_t nphi, Scalar* doublings2D) {
    // map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128 if spin == 0 else ctype[map.dtype])
    //idx is nrings
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t nphihalf = nphi / 2;
    if (idx <= nring) {
        // map_dfs[:ntheta, :] = map[0]
        for (int i = 0; i < nring; i++) {
            for (int j = 0; j < nphi; j++) {
                doublings2D[i][j] = synth2D[i][j];
            }
        }
        // map_dfs[ntheta:, :nphihalf] = map_dfs[ntheta - 2:0:-1, nphihalf:]
        // map_dfs[ntheta:, nphihalf:] = map_dfs[ntheta - 2:0:-1, :nphihalf]
        for (int i = nring; i < 2*nring; i++) {
            for (int j = 0; j < nphihalf; j++) {
                doublings2D[i][j] = synth2D[nring-i][nphihalf + j];
                doublings2D[i][nphihalf + j] = synth2D[nring-i][j];
            }
        }
    }
}

template <typename Scalar>
void CUdoubling_2D(
    nb::ndarray<const Scalar, nb::ndim<2>, nb::device::cuda> synth2D,
    const size_t nring,
    const size_t nphi,
    nb::ndarray<Scalar, nb::ndim<2>, nb::device::cuda> outarr_doubling2D) {

    const int threadsPerBlock = 256;
    int blocksPerGrid = (nring + threadsPerBlock - 1) / threadsPerBlock;
    compute_doubling_spin0_2D<<<blocksPerGrid, threadsPerBlock>>>(synth2D.data(), nring, nphi, outarr_doubling2D.data());
    cudaDeviceSynchronize();
}


template <typename Scalar>
__global__ void compute_adjoint_doubling_spin0_1D(const Scalar* doubling1D, const size_t ntheta, const size_t nphi, Scalar* synth1D) {
    // ntheta here is undoubled, idx goes across ntheta-1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= ntheta-1) {
        const size_t nphihalf = nphi / 2;
        const size_t npixplusnphihalf = 2*(ntheta-2)*nphi + nphihalf;
        const size_t doubleringi =  (ntheta-1 + idx) % ntheta;
        for (size_t phii = 0; phii < nphi; ++phii) {
            synth1D[idx * nphi + phii] = doubling1D[idx * nphi + phii];
            if (idx>0 and idx<ntheta-1){
                if (phii<nphihalf){
                    synth1D[idx * nphi + phii] += doubling1D[npixplusnphihalf - (doubleringi-1)*nphi + phii];
                } else {
                    synth1D[idx * nphi + phii] += doubling1D[npixplusnphihalf - (doubleringi)*nphi + phii];
                }
            }
        }
    }
}


template <typename Scalar>
void CUadjoint_doubling_1D(
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> synth1D,
    const size_t nring,
    const size_t nphi,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_doubling1D) {

    const int threadsPerBlock = 256;
    int blocksPerGrid = (nring + threadsPerBlock - 1) / threadsPerBlock;
    compute_adjoint_doubling_spin0_1D<<<blocksPerGrid, threadsPerBlock>>>(synth1D.data(), nring, nphi, outarr_doubling1D.data());
    cudaDeviceSynchronize();
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}


template <typename Scalar>
__global__ void compute_doubling_spin0_1D(const Scalar* synth1D, const size_t ntheta, const size_t nphi, Scalar* doubling1D) {
    //idx is ntheta of doubled map (idx = 2*ntheta-2)
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nphihalf = nphi / 2;
    const size_t npixplusnphihalf = ntheta*nphi + nphihalf;
    if (idx < ntheta) {
        if (idx < (ntheta+2)/2) {
            for (int j = 0; j < nphi; j++) {
                doubling1D[nphi*idx+j] = synth1D[nphi*idx+j];
            }
        }
        if (idx >= (ntheta+2)/2) {
            for (size_t j = 0; j < nphi; j++) {
                if (j<nphihalf) {
                    doubling1D[nphi*idx+j] = synth1D[npixplusnphihalf - idx*nphi + j];
                } else {
                    doubling1D[nphi*idx+j] = synth1D[npixplusnphihalf - (idx+1)*nphi + j];
                }
            }
        }
    }
}

template <typename Scalar>
void CUdoubling_1D(
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> synth1D,
    const size_t nring,
    const size_t nphi,
    nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_doubling1D) {

    const int threadsPerBlock = 256;
    int blocksPerGrid = (nring + threadsPerBlock - 1) / threadsPerBlock;
    compute_doubling_spin0_1D<<<blocksPerGrid, threadsPerBlock>>>(synth1D.data(), nring, nphi, outarr_doubling1D.data());
    cudaDeviceSynchronize();
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

NB_MODULE(dopy, m) {
    // m.def("CUdoubling_2D",
    //     &CUdoubling_2D<double>,
    //     "synth2D"_a.noconvert(),
    //     "nring"_a.noconvert(),
    //     "nphi"_a.noconvert(),
    //     "outarr_doubling2D"_a.noconvert()
    // );
    // m.def("CUdoubling_2Dto1D",
    //     &CUdoubling_cparr1D<double>,
    //     "synth1D"_a.noconvert(),
    //     "nring"_a.noconvert(),
    //     "nphi"_a.noconvert(),
    //     "outarr_doubling1D"_a.noconvert()
    // );
    m.def("CUdoubling_1D",
        &CUdoubling_1D<double>,
        "synth1D"_a.noconvert(),
        "nring"_a.noconvert(),
        "nphi"_a.noconvert(),
        "outarr_doubling1D"_a.noconvert()
    );
    m.def("CUadjoint_doubling_1D",
        &CUadjoint_doubling_1D<double>,
        "synth1D"_a.noconvert(),
        "nring"_a.noconvert(),
        "nphi"_a.noconvert(),
        "outarr_adjoint_doubling1D"_a.noconvert()
    );
}