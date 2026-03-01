#include "naive_gemm_omp.h"

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
    const std::vector<float>& b,
    int n) {

    std::vector<float> c(n * n);

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int i_mult_n = i * n;
            int k_mult_n = k * n;
#pragma omp simd
            for (int j = 0; j < n - 7; j += 8) {
                c[i_mult_n + j] += a[i_mult_n + k] * b[k_mult_n + j];
                c[i_mult_n + (j + 1)] += a[i_mult_n + k] * b[k_mult_n + (j + 1)];
                c[i_mult_n + (j + 2)] += a[i_mult_n + k] * b[k_mult_n + (j + 2)];
                c[i_mult_n + (j + 3)] += a[i_mult_n + k] * b[k_mult_n + (j + 3)];
                c[i_mult_n + (j + 4)] += a[i_mult_n + k] * b[k_mult_n + (j + 4)];
                c[i_mult_n + (j + 5)] += a[i_mult_n + k] * b[k_mult_n + (j + 5)];
                c[i_mult_n + (j + 6)] += a[i_mult_n + k] * b[k_mult_n + (j + 6)];
                c[i_mult_n + (j + 7)] += a[i_mult_n + k] * b[k_mult_n + (j + 7)];
            }
            
            // хвосты
            int ostalos = n % 8;
            for (int j = n - ostalos; j < n; j++) {
                c[i_mult_n + j] += a[i_mult_n + k] * b[k_mult_n + j];
            }
        }
    }

    return c;
}