#include "gelu_cuda.h"

#include <cuda_runtime.h>
#include <cmath>

__global__ void kernel(int n, float* d_memory) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float x = d_memory[idx];

        float _2y = x * 1.595769122f * (1.0f + 0.044715f * x * x);
        float e2y = expf(_2y);
        float res = x * (1.0f - 1.0f / (e2y + 1.0f));

        d_memory[idx] = res;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {

    int n = input.size();

    std::vector<float> output(n);

    const float* input_ptr = input.data();
    float* output_ptr = output.data();

    float* d_memory;
    
    cudaMalloc(&d_memory, n * sizeof(float));

    cudaMemcpy(d_memory, input_ptr, n * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int blocks_size = 32;

    kernel<<<(n + blocks_size - 1) / blocks_size, blocks_size>>>(n, d_memory);

    cudaMemcpy(output_ptr, d_memory, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_memory);

    return output;
}