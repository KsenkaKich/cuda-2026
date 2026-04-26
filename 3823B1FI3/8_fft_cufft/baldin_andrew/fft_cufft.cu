#include "fft_cufft.h"
#include <cuda_runtime.h>
#include <cufft.h>

__global__ void normalize_kernel(cufftComplex* __restrict__ data, int complex, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < complex) {
        data[i].x *= scale;
        data[i].y *= scale;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) {
    int size = input.size();
    int complex = size / 2;
    int n = complex / batch;

    size_t bytes = size * sizeof(float);

    cufftComplex* d_data;
    cudaMalloc(&d_data, bytes);

    cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_C2C, batch);

    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);

    int block_size = 256;
    int grid_size = (complex + block_size - 1) / block_size;
    float scale = 1.0f / static_cast<float>(n);

    normalize_kernel <<< grid_size, block_size >>> (d_data, complex, scale);

    std::vector<float> output(size);
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(plan);
    cudaFree(d_data);

    return output;
}
