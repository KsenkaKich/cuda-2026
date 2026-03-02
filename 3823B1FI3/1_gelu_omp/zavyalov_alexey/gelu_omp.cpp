#include "gelu_omp.h"

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> res(input.size());
    int n = input.size();
    int i = 0;
    float sqrt_2_div_pi = 0.7978845608028653558798921198687637369517172623298693153318516593f;

#pragma omp parallel for
    for (i = 0; i < n - 7; i += 8) {

        res[i] = input[i] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i] + 0.044715f * input[i] * input[i] * input[i])) + 1.0f));
        res[i + 1] = input[i + 1] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i + 1] + 0.044715f * input[i + 1] * input[i + 1] * input[i + 1])) + 1.0f));
        res[i + 2] = input[i + 2] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i + 2] + 0.044715f * input[i + 2] * input[i + 2] * input[i + 2])) + 1.0f));
        res[i + 3] = input[i + 3] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i + 3] + 0.044715f * input[i + 3] * input[i + 3] * input[i + 3])) + 1.0f));
        res[i + 4] = input[i + 4] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i + 4] + 0.044715f * input[i + 4] * input[i + 4] * input[i + 4])) + 1.0f));
        res[i + 5] = input[i + 5] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i + 5] + 0.044715f * input[i + 5] * input[i + 5] * input[i + 5])) + 1.0f));
        res[i + 6] = input[i + 6] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i + 6] + 0.044715f * input[i + 6] * input[i + 6] * input[i + 6])) + 1.0f));
        res[i + 7] = input[i + 7] * (1.0f - 1.0f / (exp(2.0f * sqrt_2_div_pi * (input[i + 7] + 0.044715f * input[i + 7] * input[i + 7] * input[i + 7])) + 1.0f));
    }

    int ostalos = n % 8;
    for (int i = std::max(0, n - ostalos); i < n; i++) {
        float x = input[i];
        float tanh_arg = sqrt_2_div_pi * (x + 0.044715f * x * x * x);
        float exp_precalc = exp(2.0f * tanh_arg);
        float cnst = (1.0f - 1.0f / (exp_precalc + 1.0f));
        res[i] = x * cnst;
    }
    return res;
}