#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int deviceId = 0; // Device ID (0 for the first GPU)

    // Get device properties
    cudaError_t error = cudaGetDeviceProperties(&prop, deviceId);
    if (error != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Print device properties
    printf("Device Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %zu bytes\n", prop.totalGlobalMem);
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Blocks per Grid (x): %d\n", prop.maxGridSize[0]);
    printf("Max Blocks per Grid (y): %d\n", prop.maxGridSize[1]);
    printf("Max Blocks per Grid (z): %d\n", prop.maxGridSize[2]);

    return 0;
}