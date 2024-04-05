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
void CUdoubling_cparr2D(
    nb::ndarray<const Scalar, nb::ndim<2>, nb::device::cuda> synth2D,
    const size_t nring,
    const size_t nphi,
    nb::ndarray<Scalar, nb::ndim<2>, nb::device::cuda> outarr_doubling2D) {

    const int threadsPerBlock = 256;
    int blocksPerGrid = (nring + threadsPerBlock - 1) / threadsPerBlock;
    compute_doubling_spin0_2D<<<blocksPerGrid, threadsPerBlock>>>(synth2D.data(), nring, nphi, outarr_doubling2D.data());
    cudaDeviceSynchronize();
}


// template <typename Scalar>
// __global__ void compute_doubling_spin0_2Dto1D(const Scalar* synth1D, const size_t ntheta, const size_t nphi, Scalar* doubling1D) {
//     // map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128 if spin == 0 else ctype[map.dtype])
//     //idx is ntheta
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     const size_t nphihalf = nphi / 2;
//     if (idx <= ntheta) {
//         // map_dfs[:ntheta, :] = map[0]
//         for (int i = 0; i < ntheta; i++) {
//             for (int j = 0; j < nphi; j++) {
//                 doubling1D[i * nphi + j] = synth1D[i * nphi + j];
//             }
//         }
//         for (int i = ntheta; i < 2 * ntheta - 2; i++) {
//             for (int j = 0; j < nphihalf; j++) {
//                 doubling1D[i * nphi + j] = synth1D[(2 * ntheta - 3 - i) * nphi + (nphi - nphihalf) + j];
//             }
//         }
//         for (int i = ntheta; i < 2 * ntheta - 2; i++) {
//             for (int j = nphihalf; j < nphi; j++) {
//                 doubling1D[i * nphi + j] = synth1D[(2 * ntheta - 3 - i) * nphi + (j - nphihalf)];
//             }
//         }
//     }
// }

// template <typename Scalar>
// void CUdoubling_2Dto1D(
//     nb::ndarray<Scalar, nb::ndim<2>, nb::device::cuda> synth1D,
//     const size_t nring,
//     const size_t nphi,
//     nb::ndarray<Scalar, nb::ndim<1>, nb::device::cuda> outarr_doubling1D) {

//     const int threadsPerBlock = 256;
//     int blocksPerGrid = (nring + threadsPerBlock - 1) / threadsPerBlock;
//     compute_doubling_spin0_1D<<<blocksPerGrid, threadsPerBlock>>>(synth1D.data(), nring, nphi, outarr_doubling1D.data());
//     cudaDeviceSynchronize();
// }

template <typename Scalar>
__global__ void compute_doubling_spin0_1D(const Scalar* synth1D, const size_t ntheta, const size_t nphi, Scalar* doubling1D) {
    // map_dfs = np.empty((2 * ntheta - 2, nphi), dtype=np.complex128 if spin == 0 else ctype[map.dtype])
    //idx is ntheta
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t nphihalf = nphi / 2;
    if (idx < ntheta) {
        // map_dfs[:ntheta, :] = map[0]
        for (int j = 0; j < nphi; j++) {
                doubling1D[nphi*idx+j] = synth1D[nphi*idx+j];
                // doubling1D[2*nphi*idx + j] = synth1D[nphi-j];
                // doubling1D[2*nphi*idx + j] = 2.*synth1D[j];
        }
        for (int j = 0; j < nphi/2; j++) {
            doubling1D[2*nphi*idx+j] = synth1D[nphi*idx+j];
        }
        for (int j = nphi/2; j < nphi; j++) {
            doubling1D[2*nphi*idx+j] = synth1D[nphi*idx+j-nphi/2];
        }
    }
}


template <typename Scalar>
void CUdoubling_cparr1D(
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
    // m.def("CUdoubling_cparr2D",
    //     &CUdoubling_cparr2D<double>,
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
    m.def("CUdoubling_cparr1D",
        &CUdoubling_cparr1D<double>,
        "synth1D"_a.noconvert(),
        "nring"_a.noconvert(),
        "nphi"_a.noconvert(),
        "outarr_doubling1D"_a.noconvert()
    );
}