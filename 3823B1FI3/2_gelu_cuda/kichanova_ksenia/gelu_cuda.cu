#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <cmath>

#define SQRT_2_OVER_PI 0.7978845608028654f
#define COEFF 0.044715f

__global__ void gelu_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx * 2; i < n - 1; i += stride * 2) {
        const float x0 = input[i];
        const float x1 = input[i + 1];
        
        const float x0_3 = x0 * x0 * x0;
        const float x1_3 = x1 * x1 * x1;
        
        const float p0 = SQRT_2_OVER_PI * (x0 + COEFF * x0_3);
        const float p1 = SQRT_2_OVER_PI * (x1 + COEFF * x1_3);
        
        const float exp_2p0 = __expf(2.0f * p0);
        const float exp_2p1 = __expf(2.0f * p1);
        
        output[i] = x0 * exp_2p0 / (exp_2p0 + 1.0f);
        output[i + 1] = x1 * exp_2p1 / (exp_2p1 + 1.0f);
    }
    
    if (n % 2 == 1 && idx == 0) {
        const size_t i = n - 1;
        const float x = input[i];
        const float p = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        const float exp_2p = __expf(2.0f * p);
        output[i] = x * exp_2p / (exp_2p + 1.0f);
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    const size_t bytes = n * sizeof(float);
    
    static float *d_input = nullptr;
    static float *d_output = nullptr;
    static size_t allocated = 0;
    static cudaStream_t stream = nullptr;
    
    if (!stream) {
        cudaStreamCreate(&stream);
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        allocated = n;
    }
    
    if (allocated < n) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaMalloc(&d_input, bytes);
        cudaMalloc(&d_output, bytes);
        allocated = n;
    }
    
    std::vector<float> result(n);
    
    cudaMemcpyAsync(d_input, input.data(), bytes, cudaMemcpyHostToDevice, stream);
    
    const int block_size = 256;
    const int grid_size = 24 * 16;
    
    gelu_kernel<<<grid_size, block_size, 0, stream>>>(d_input, d_output, n);
    
    cudaMemcpyAsync(result.data(), d_output, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return result;
}