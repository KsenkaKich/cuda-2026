#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

// ядро для вычисления произведения матриц A и B с сохранением результата в C на каждом ядре GPU
__global__ void NaiveGemmKernel(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // индекс строки в C
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // индекс столбца в C (шаг 4 для float)

    if (row < n && col < n) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f); 

        for (int k = 0; k < n; ++k) {
            float a_val = A[row * n + k];

            float4 b_val = ((float4*)B)[(k * n + col) / 4];

            // накопление результата
            sum.x += a_val * b_val.x; 
            sum.y += a_val * b_val.y;
            sum.z += a_val * b_val.z;
            sum.w += a_val * b_val.w;
        }

        // запись сразу 4 результатов в C
        ((float4*)C)[(row * n + col) / 4] = sum;
    }
}

// основная функция (выполняется на CPU)
std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, 
                                 const std::vector<float>& b, 
                                 int n) {

    size_t size = (size_t)n * n;

    // статическая указатели, чтобы не делать аллокацию и деаллокацию памяти на каждом вызове 
    static float *d_a = nullptr;
    static float *d_b = nullptr;
    static float *d_c = nullptr;
    static int allocated_size = 0;
    static cudaStream_t stream = nullptr;

    // выделение памяти на GPU
    if (allocated_size < n) {
        if (d_a)
            cudaFree(d_a);
        if (d_b)
            cudaFree(d_b);
        if (d_c)
            cudaFree(d_c);
        cudaMalloc(&d_a, size * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_c, size * sizeof(float));
        if (!stream)
            cudaStreamCreate(&stream); // создание потока для асинхронных операций
        allocated_size = n;
    }

    // асинхронное копирование входных матриц с CPU на GPU в потоке stream
    cudaMemcpyAsync(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // настройки сетки
    dim3 threads(16, 16); // 256 потоков на блок
    dim3 blocks((n / 4 + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y); // количество блоков для покрытия всей матрицы

    NaiveGemmKernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_c, n); // запуск ядра на GPU с конфигурацией запуска асинхронно в потоке stream

    std::vector<float> c(size); // пока GPU выполняет вычисления, выделяем память для результата на CPU

    cudaMemcpyAsync(c.data(), d_c, size * sizeof(float), cudaMemcpyDeviceToHost, stream); // асинхронное копирование результата с GPU на CPU в потоке stream

    cudaStreamSynchronize(stream); // синхронизация всех операций в потоке stream

    return c;
}