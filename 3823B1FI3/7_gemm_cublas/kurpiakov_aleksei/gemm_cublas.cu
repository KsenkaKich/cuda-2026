#include "gemm_cublas.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
                                
    const int bytes = static_cast<int>(a.size() * sizeof(float));
    float *A;
    float *B;
    float *res;

    cudaMalloc(&A, bytes);
    cudaMalloc(&B, bytes);
    cudaMalloc(&res, bytes);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, A, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, B, n);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   n, n, n,
                   &alpha,
                   B, n,
                   A, n,
                   &beta,
                   res, n);

    std::vector<float> result(a.size());
    cublasGetMatrix(n, n, sizeof(float), res, n, result.data(), n);

    cublasDestroy_v2(handle);
    cudaFree(A);
    cudaFree(B);
    cudaFree(res);

    return result;
}