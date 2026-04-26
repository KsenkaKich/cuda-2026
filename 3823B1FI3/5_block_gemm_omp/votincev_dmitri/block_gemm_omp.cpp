#include "naive_gemm_omp.h"
#include "omp.h"
#include <vector>

// GEMM - general matrix multiplication

/*
Ключи в Visual Studio:
/openmp (/openmp:llvm) - многопоточность
/fp:fast вместо /fp:precise - быстрые floating point (expf становится быстрее)
/arch:AVX2 - векторные инструкции (в регистре одновременно 8 float)
AVX2 регстры: 256 бит; float: 32 бит; 256/32 = 8
/GL - глобальная оптимизация
/O2  - агрессивная оптимизация
/openmp:experimental - для simd (векторизации)
*/

std::vector<float> BlockGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> answer(n * n, 0.0f);


    // блочное умножение 
    const int bl_size = 64;
    const int bl_count = n / bl_size;

    // распараллеливаем внешние блоки
#pragma omp parallel for collapse(2)
    for (int bl_j = 0; bl_j < bl_count; bl_j++) {
        for (int bl_i = 0; bl_i < bl_count; bl_i++) {
            for (int bl_k = 0; bl_k < bl_count; bl_k++) {
                for (int j = bl_j * bl_size; j < (bl_j + 1) * bl_size; j++) {
                    float* res_row = &answer[j * n];
                    const float* a_row = &a[j * n];
                    for (int k = bl_k * bl_size; k < (bl_k + 1) * bl_size; k++) {
                        float a_val = a_row[k];
                        const float* b_row = &b[k * n];



#pragma omp simd
                        for (int i = bl_i * bl_size; i < (bl_i + 1) * bl_size; ++i) {
                            res_row[i] += a_val * b_row[i];
                        }
                    }
                }
            }
        }
    }

    return answer;
}

