#pragma GCC target("avx2")
#include "block_gemm_omp.h"

#include <algorithm>

#pragma GCC optimize("unroll-loops")
std::vector<float> BlockGemmOMP(const std::vector<float> &a,
                                const std::vector<float> &b, int n) {

  const int BLOCK_SIZE = 64;
  const int BLOCKS_CNT = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  std::vector<float> res(n * n);

#pragma omp parallel for schedule(static)
  for (int ii = 0; ii < BLOCKS_CNT; ii++) {
    int i_start = ii * BLOCK_SIZE;
    int i_end = std::min(i_start + BLOCK_SIZE, n);
    for (int jj = 0; jj < BLOCKS_CNT; jj++) {
      int j_start = jj * BLOCK_SIZE;
      int j_end = std::min(j_start + BLOCK_SIZE, n);
      for (int kk = 0; kk < BLOCKS_CNT; kk++) {
        int k_start = kk * BLOCK_SIZE;
        int k_end = std::min(k_start + BLOCK_SIZE, n);

        for (int i = i_start; i < i_end; i++) {
          for (int k = k_start; k < k_end; k++) {
            float aik = a[i * n + k];
            int in = i * n;
            int kn = k * n;
#pragma omp simd
            for (int j = j_start; j < j_end; j++) {
              res[in + j] += aik * b[kn + j];
            }
          }
        }
      }
    }
  }

  return res;
}