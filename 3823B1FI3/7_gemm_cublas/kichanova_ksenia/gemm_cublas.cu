#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

static float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
static size_t allocated_a = 0, allocated_b = 0, allocated_c = 0;
static cublasHandle_t handle = nullptr;
static bool initialized = false;

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    const size_t bytes = n * n * sizeof(float);
    
    if (!initialized) {
        cublasCreate(&handle);
        initialized = true;
    }
    
    if (allocated_a < bytes) {
        if (d_a) cudaFree(d_a);
        cudaMalloc(&d_a, bytes);
        allocated_a = bytes;
    }
    if (allocated_b < bytes) {
        if (d_b) cudaFree(d_b);
        cudaMalloc(&d_b, bytes);
        allocated_b = bytes;
    }
    if (allocated_c < bytes) {
        if (d_c) cudaFree(d_c);
        cudaMalloc(&d_c, bytes);
        allocated_c = bytes;
    }
    
    cublasSetMatrix(n, n, sizeof(float), a.data(), n, d_a, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, d_b, n);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n, &alpha, d_b, n, d_a, n, &beta, d_c, n);
    
    std::vector<float> c(n * n);
    cublasGetMatrix(n, n, sizeof(float), d_c, n, c.data(), n);
    
    return c;
}