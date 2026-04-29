#include "gelu_omp.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> result(n);
    const float sqrt_2_over_pi = std::sqrt(2.0f / static_cast<float>(M_PI));
    const float coeff = 0.044715f;
    const float half = 0.5f;
    const float one = 1.0f;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const float x = input[i];
        const float x3 = x * x * x;
        const float inner = sqrt_2_over_pi * (x + coeff * x3);
        const float tanh_val = tanhf(inner);   
        result[i] = half * x * (one + tanh_val);
    }
    return result;
}