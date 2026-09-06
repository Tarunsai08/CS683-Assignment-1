// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    for (int i = 0; i < M; ++i) {
        const float* a_row = A + (long)i * lda;
        for (int j = 0; j < N; ++j) {
            const float* b_row = B + (long)j * ldb;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            for(; k + 8 <= K; k+=8){
                __m256 a_vec = _mm256_loadu_ps(&a_row[k]);
                __m256 b_vec = _mm256_loadu_ps(&b_row[k]);

                acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
                _mm256_storeu_ps(&C[i * ldc + j], acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);

            float sum = _mm_cvtss_f32(sum128);
            for (; k < K; ++k) {
                sum += a_row[k] * b_row[k];
            }
            C[(long)i * ldc + j] = sum;
        }
    }
}
