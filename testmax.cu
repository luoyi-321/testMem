#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <map>
#define EPSILON 1.0e-15
#define NUM_BINS 2048

#include <chrono>

// #include <windows.h> 
// #include "mat.h"


#include <fstream>
#include <sstream>

#include <cuComplex.h> 
#include <cufft.h>

#include <thrust/complex.h>

#define M_PI 3.141592653589793
#define blockSize 1024
#define TARGETNUM 4194304

using namespace std;

__device__ double atomicMaxDouble(double *address, double value) {
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(max(value, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void findMaxComplexKernel(const cuDoubleComplex *data, double *maxReal, double *maxImag, int size) {
    __shared__ double sharedMaxReal;
    __shared__ double sharedMaxImag;

    // Initialize shared memory
    if (threadIdx.x == 0) {
        sharedMaxReal = -DBL_MAX;
        sharedMaxImag = -DBL_MAX;
    }
    __syncthreads();

    // Each thread processes one element
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        double realPart = abs(cuCreal(data[tid]));
        double imagPart = abs(cuCimag(data[tid]));

        // Update shared memory with the max values
        atomicMaxDouble(&sharedMaxReal, realPart);
        atomicMaxDouble(&sharedMaxImag, imagPart);
    }
    __syncthreads();

    // Write the result from shared memory to global memory
    if (threadIdx.x == 0) {
        atomicMaxDouble(maxReal, sharedMaxReal);
        atomicMaxDouble(maxImag, sharedMaxImag);
    }
    // printf("maxImag: %f \n",maxImag);

}

void findMaxComplex(const cuDoubleComplex *data, int size, double &maxReal, double &maxImag) {
    cuDoubleComplex *d_data;
    double *d_maxReal, *d_maxImag;
    double h_maxReal = -DBL_MAX;
    double h_maxImag = -DBL_MAX;

    // Allocate device memory
    cudaMalloc((void**)&d_data, size * sizeof(cuDoubleComplex));
    cudaMalloc((void**)&d_maxReal, sizeof(double));
    cudaMalloc((void**)&d_maxImag, sizeof(double));

    
    // Copy data from host to device
    cudaMemcpy(d_data, data, size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemset(d_maxReal,-DBL_MAX,sizeof(double));
    cudaMemset(d_maxImag,-DBL_MAX,sizeof(double));
    // cudaMemcpy(d_maxReal, &h_maxReal, sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_maxImag, &h_maxImag, sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    findMaxComplexKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_maxReal, d_maxImag, size);
    printf("d_maxReal: %f \n",d_maxReal);
    // Copy results from device to host
    cudaMemcpy(&maxReal, d_maxReal, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxImag, d_maxImag, sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_maxReal);
    cudaFree(d_maxImag);
}

int main() {
    const int size = 10;
    cuDoubleComplex h_data[size] = {
        make_cuDoubleComplex(3.0, 5.0),
        make_cuDoubleComplex(7.0, 2.0),
        make_cuDoubleComplex(8.0, 9.0),
        make_cuDoubleComplex(4.0, 1.0),
        make_cuDoubleComplex(6.0, 10.0),
        make_cuDoubleComplex(0.0, -13.0),
        make_cuDoubleComplex(-2.0, 4.0),
        make_cuDoubleComplex(9.0, -1.0),
        make_cuDoubleComplex(-5.0, 6.0),
        make_cuDoubleComplex(1.0, -2.0)
    };

    double maxReal, maxImag;
    findMaxComplex(h_data, size, maxReal, maxImag);
    printf("Max real part: %f\n", maxReal);
    printf("Max imaginary part: %f\n", maxImag);

    return 0;
}
