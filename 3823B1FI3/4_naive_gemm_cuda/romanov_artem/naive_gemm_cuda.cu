#include "naive_gemm_cuda.h"

__global__ void NaiveGemmKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = size_t(n) * n * sizeof(float);

    float* dA;
    cudaMalloc(&dA, bytes);

    float* dB;
    cudaMalloc(&dB, bytes);

    float* dC;
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    constexpr int BS = 16;
    dim3 block(BS, BS);
    dim3 grid((n + BS - 1) / BS, (n + BS - 1) / BS);

    NaiveGemmKernel << <grid, block >> > (dA, dB, dC, n);
    cudaDeviceSynchronize();

    std::vector<float> result(size_t(n) * n);
    cudaMemcpy(result.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return result;
}