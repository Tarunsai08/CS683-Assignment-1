// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

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

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 256;

    for (int ii = 0; ii < M; ii += BLOCK_M) {
        const int i_end = (ii + BLOCK_M < M) ? ii + BLOCK_M : M;

        for (int jj = 0; jj < N; jj += BLOCK_N) {
            const int j_end = (jj + BLOCK_N < N) ? jj + BLOCK_N : N;

            for (int i = ii; i + 3 < i_end; i += 4) {

                for (int j = jj; j + 1 < j_end; j += 2) {

                    float c00 = 0.0f;
                    float c01 = 0.0f;
                    float c10 = 0.0f;
                    float c11 = 0.0f;
                    float c20 = 0.0f;
                    float c21 = 0.0f;
                    float c30 = 0.0f;
                    float c31 = 0.0f;

                    for (int kk = 0; kk < K; kk += BLOCK_K) {
                        const int k_end =
                            (kk + BLOCK_K < K) ? kk + BLOCK_K : K;

                        __m256 v00 = _mm256_setzero_ps();
                        __m256 v01 = _mm256_setzero_ps();
                        __m256 v10 = _mm256_setzero_ps();
                        __m256 v11 = _mm256_setzero_ps();
                        __m256 v20 = _mm256_setzero_ps();
                        __m256 v21 = _mm256_setzero_ps();
                        __m256 v30 = _mm256_setzero_ps();
                        __m256 v31 = _mm256_setzero_ps();

                        int k = kk;

                        for (; k + 7 < k_end; k += 8) {
                            const __m256 b0 =
                                _mm256_loadu_ps(B + j * ldb + k);
                            const __m256 b1 =
                                _mm256_loadu_ps(B + (j + 1) * ldb + k);

                            const __m256 a0 =
                                _mm256_loadu_ps(A + i * lda + k);
                            const __m256 a1 =
                                _mm256_loadu_ps(A + (i + 1) * lda + k);
                            const __m256 a2 =
                                _mm256_loadu_ps(A + (i + 2) * lda + k);
                            const __m256 a3 =
                                _mm256_loadu_ps(A + (i + 3) * lda + k);

                            v00 = _mm256_fmadd_ps(a0, b0, v00);
                            v01 = _mm256_fmadd_ps(a0, b1, v01);

                            v10 = _mm256_fmadd_ps(a1, b0, v10);
                            v11 = _mm256_fmadd_ps(a1, b1, v11);

                            v20 = _mm256_fmadd_ps(a2, b0, v20);
                            v21 = _mm256_fmadd_ps(a2, b1, v21);

                            v30 = _mm256_fmadd_ps(a3, b0, v30);
                            v31 = _mm256_fmadd_ps(a3, b1, v31);
                        }

                        c00 += hsum256(v00);
                        c01 += hsum256(v01);
                        c10 += hsum256(v10);
                        c11 += hsum256(v11);
                        c20 += hsum256(v20);
                        c21 += hsum256(v21);
                        c30 += hsum256(v30);
                        c31 += hsum256(v31);

                        for (; k < k_end; ++k) {
                            const float b0 = B[j * ldb + k];
                            const float b1 = B[(j + 1) * ldb + k];

                            const float a0 = A[i * lda + k];
                            const float a1 = A[(i + 1) * lda + k];
                            const float a2 = A[(i + 2) * lda + k];
                            const float a3 = A[(i + 3) * lda + k];

                            c00 += a0 * b0;
                            c01 += a0 * b1;

                            c10 += a1 * b0;
                            c11 += a1 * b1;

                            c20 += a2 * b0;
                            c21 += a2 * b1;

                            c30 += a3 * b0;
                            c31 += a3 * b1;
                        }
                    }

                    C[i * ldc + j] = c00;
                    C[i * ldc + j + 1] = c01;

                    C[(i + 1) * ldc + j] = c10;
                    C[(i + 1) * ldc + j + 1] = c11;

                    C[(i + 2) * ldc + j] = c20;
                    C[(i + 2) * ldc + j + 1] = c21;

                    C[(i + 3) * ldc + j] = c30;
                    C[(i + 3) * ldc + j + 1] = c31;
                }

                for (int j = (j_end & ~1); j < j_end; ++j) {
                    for (int r = i; r < i + 4; ++r) {
                        float sum = 0.0f;

                        for (int k = 0; k < K; ++k)
                            sum += A[r * lda + k] *
                                   B[j * ldb + k];

                        C[r * ldc + j] = sum;
                    }
                }
            }

            for (int i = (i_end & ~3); i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    float sum = 0.0f;

                    for (int k = 0; k < K; ++k)
                        sum += A[i * lda + k] *
                               B[j * ldb + k];

                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}