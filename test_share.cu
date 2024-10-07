#define _USE_MATH_DEFINES

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

#define TILE_SIZE 32  // 假设我们采用 32x32 的块大小
#define NRN 2048      // 假设 NRN 是 2048

struct arae_data {
    double d_arae_x[NRN];
    double d_arae_y[NRN];
    double d_arae_RCS[NRN];
};

__global__ void printKernel(cuDoubleComplex *xx_match,int size){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx < size){
        printf("xx_match:[%d].x : %f\n",idx,xx_match[idx].x);
    }
}

__global__ void transferDataToMatch(arae_data *d_area_data, cuDoubleComplex *xx_match, int N) {
    // 共享内存，用来暂存 d_area_data 数据
    __shared__ double shared_x[TILE_SIZE][TILE_SIZE];
    __shared__ double shared_y[TILE_SIZE][TILE_SIZE];
    // __shared__ double shared_rcs[TILE_SsIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // 从全局内存读取 arae_data 的数据到共享内存
    if (row < N && col < N) {
        shared_x[ty][tx] = d_area_data->d_arae_x[row * N + col];
        shared_y[ty][tx] = d_area_data->d_arae_y[row * N + col];
    }
    if(row<N && col<N){
        
    }
    // 确保所有线程都完成读取
    __syncthreads();

    // 将共享内存中的值赋给 xx_match 矩阵
    if (row < N && col < N) {
        // 这里你可以根据具体需求组合 d_arae_x、d_arae_y 和 d_arae_RCS 的值
        xx_match[row * N + col] = make_cuDoubleComplex(shared_x[ty][tx], shared_y[ty][tx]);
    }
}

int main() {
    int N = 2048;  // 假设矩阵大小为 2048x2048

    // 分配全局内存
    arae_data *d_area_data = NULL;
    cudaMalloc(&d_area_data, N * sizeof(arae_data));

    // 分配 xx_match 的全局内存
    cuDoubleComplex *xx_match;
    cudaMalloc(&xx_match, N * N * sizeof(cuDoubleComplex));

    // 定义 grid 和 block 大小
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // 调用 kernel
    transferDataToMatch<<<gridSize, blockSize>>>(d_area_data, xx_match, N);
    // printKernel<<<gridSize,blockSize>>>(xx_match,N*N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // 清理
    cudaFree(d_area_data);
    cudaFree(xx_match);

    return 0;
}
