#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = sizeof(float) * n * n;

    float* dA;
    cudaMalloc(&dA, bytes);

    float* dB;
    cudaMalloc(&dB, bytes);

    float* dC;
    cudaMalloc(&dC, bytes);

    cudaMemcpy(dA, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b.data(), bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        dB, n,
        dA, n,
        &beta,
        dC, n);

    std::vector<float> result(n * n);
    cudaMemcpy(result.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return result;
}