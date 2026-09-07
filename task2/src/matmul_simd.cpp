// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

inline float hsum256(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum  = _mm_add_ps(vlow, vhigh);

    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);

    return _mm_cvtss_f32(vsum);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            
            __m256 acc_vec = _mm256_setzero_ps();

            for (int p = 0; p < K; p += 8) {
                __m256 vecA = _mm256_loadu_ps(&A[i * lda + p]);
                __m256 vecB = _mm256_loadu_ps(&B[j * ldb + p]);

                acc_vec = _mm256_fmadd_ps(vecA, vecB, acc_vec);
            }

            C[i * ldc + j] = hsum256(acc_vec);
        }
    }
}