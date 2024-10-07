# CUDA Row-wise Reduction

This project implements a row-wise reduction operation on a 2048x2048 matrix using CUDA. Each row of the matrix is reduced by summing all its elements, with the final result stored in the first column of each row.

## Project Overview

The main CUDA kernel (`reduceRows`) performs the following tasks:
- Each row of the matrix is processed by a block of threads.
- The threads within each block sum the elements of the row using shared memory.
- The final sum of each row is written back to the result matrix, with the sum stored in the first element of each row.

### Key Components

1. **d_matrix**: The input 2048x2048 matrix on the device, initialized with all ones (1).
2. **d_result**: The output matrix on the device that stores the result of the row-wise sum.
3. **reduceRows kernel**: The CUDA kernel that reduces each row by summing its elements using shared memory and writes the sum back to the first element of each row.

### Parameters

- **NRN**: The size of the input matrix, which is `2048 * 2048` in this example.
- **TILE_SIZE**: Defines the size of the thread block. The default is set to 32, which is the warp size of most NVIDIA GPUs.

## Prerequisites

- CUDA Toolkit installed (tested with CUDA version 11.1 and above).
- C++ compiler (GCC or similar).
- NVIDIA GPU with compute capability supporting CUDA.

## Compilation

You can compile the project using `nvcc` (NVIDIA's CUDA compiler):

```bash
nvcc -o row_reduce row_reduce.cu
