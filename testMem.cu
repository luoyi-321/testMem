#include <cuda_runtime.h>
#include <iostream>


using namespace std;


#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error at \"" #call "\", error code mean " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    }



// CUDA 内核函数：为数组赋值
__global__ void initializeArray(double* a, int nx, int ny, int nz, double value) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // 计算一维数组中的线性索引
    if (idx < nx && idy < ny && idz < nz) {
        int index = idx + idy * nx + idz * nx * ny;
        a[index] = value;
    }
}

int main() {
    // 矩阵维度
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Shared memory per block: %d\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("Registers per thread: %d\n", prop.regsPerBlock / prop.maxThreadsPerBlock);
}
