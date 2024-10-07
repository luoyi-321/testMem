#include <cuComplex.h>
#include <cuda_runtime.h>

__global__ void initialize_batar_tr(cuFloatComplex *batar_tr, int N1, int N2) {
    // 计算线程的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // 转换为线性索引
    int index =  idy * N1 + idx;

    // 检查是否超出边界
    if (idx < N1 && idy < N2 ) {
        // 初始化为 0.0 + 0.0i
        batar_tr[index] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

int main() {
    int N1 = 2048;
    int N2 = 2048;
    // int N3 = 256;

    // 在设备端分配内存
    cuFloatComplex *d_batar_tr;
    cudaMalloc(&d_batar_tr, N1 * N2 * sizeof(cuFloatComplex));

    // 定义线程块和网格大小
    dim3 threadsPerBlock(32, 32);  // 每个块 512 个线程 (8*8*8)
    dim3 numBlocks((N1 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N2 + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 调用 CUDA 核函数
    initialize_batar_tr<<<numBlocks, threadsPerBlock>>>(d_batar_tr, N1, N2);

    // 确保所有 CUDA 任务执行完毕
    cudaDeviceSynchronize();

    // 释放设备内存
    cudaFree(d_batar_tr);

    return 0;
}
