#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

__global__ void naive_gemm_kernel(const float* __restrict__ A, 
                                  const float* __restrict__ B, 
                                  float* __restrict__ C, 
                                  int n) {
                                            
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;

        #pragma unroll
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    size_t bytes = n * n * sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block_size(16, 16); 
    dim3 grid_size((n + block_size.x - 1) / block_size.x, 
                   (n + block_size.y - 1) / block_size.y);

    naive_gemm_kernel <<< grid_size, block_size >>> (d_A, d_B, d_C, n);
    
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return c;
}