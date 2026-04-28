#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 16

__global__ void kernel(const float* A, const float* B, float* C, int n) {

    __shared__ float block_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float block_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    float sum = 0.0f;

    for (int m = 0; m < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {

        block_A[threadIdx.y][threadIdx.x] = A[row * n + (m * BLOCK_SIZE + threadIdx.x)];
        block_B[threadIdx.y][threadIdx.x] = B[(m * BLOCK_SIZE + threadIdx.y) * n + col];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += block_A[threadIdx.y][k] * block_B[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
    
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    
    float* gpu_A;
    float* gpu_B;
    float* gpu_C;
    int bytes = n * n * sizeof(float);

    cudaMalloc(&gpu_A, bytes);
    cudaMalloc(&gpu_B, bytes);
    cudaMalloc(&gpu_C, bytes);

    cudaMemcpy(gpu_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16);
    dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);
    
    kernel<<<grid_size, block_size>>>(gpu_A, gpu_B, gpu_C, n);

    std::vector<float> result(n * n);

    cudaMemcpy(result.data(), gpu_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
    return result;
}