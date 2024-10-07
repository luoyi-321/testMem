#include <cuda_runtime.h>
#include <iostream>

#define N 2048
#define NRN 2048*2048
#define TILE_SIZE 32

// CUDA Kernel for row-wise reduction using warp shuffle
__global__ void reduceRows(int* d_matrix, int* d_result) {
    __shared__ int shared[TILE_SIZE];

    int row = blockIdx.x;
    int col = threadIdx.x;

    int sum = 0;

    // Make sure the thread index is within bounds
    if (row < N && col < N) {
        // Accumulate values within the row
        for (int i = col; i < N; i += blockDim.x) {
            sum += d_matrix[row * N + i];
        }

        // Each thread in the block contributes to the shared memory
        shared[col] = sum;

        __syncthreads();

        // Perform reduction within the warp
        for (int stride = TILE_SIZE / 2; stride > 0; stride /= 2) {
            if (col < stride) {
                shared[col] += shared[col + stride];
            }
            __syncthreads();
        }

        // Store the result for the row in the first thread
        if (col == 0) {
            d_result[row] = shared[0];
        }
    }
}

int main() {


    int h_result[N];
    int *d_matrix, *d_result;

    int* h_matrix = (int*)malloc(NRN * sizeof(int));  // Allocate memory on heap

    if (h_matrix == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return -1;
    }

    // Initialize matrix with ones
    for (int i = 0; i < NRN; i++) {
        h_matrix[i] = 1;
    }

    // Allocate memory on device
    cudaError_t err = cudaMalloc(&d_matrix, N * N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (Malloc d_matrix): " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
       
    err = cudaMalloc(&d_result, N * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (Malloc d_result): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrix);
        return -1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (Memcpy h_matrix to d_matrix): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrix);
        cudaFree(d_result);
        return -1;
    }

    // Launch kernel
    reduceRows<<<N, TILE_SIZE>>>(d_matrix, d_result);

    // Check for errors after kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (Kernel Launch): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrix);
        cudaFree(d_result);
        return -1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (Memcpy d_result to h_result): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_matrix);
        cudaFree(d_result);
        return -1;
    }

    // Print the result
    for (int i = 0; i < N; i++) {
        std::cout << "Row " << i << " sum: " << h_result[i] << std::endl;
    }

    // Clean up
    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}
