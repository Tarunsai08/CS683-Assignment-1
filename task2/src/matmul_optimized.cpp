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

    // Optimal Cache Tiling parameters (L1/L2 cache fit)
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;

    for (int ii = 0; ii < M; ii += BLOCK_M) {
        const int i_end = (ii + BLOCK_M < M) ? ii + BLOCK_M : M;

        for (int jj = 0; jj < N; jj += BLOCK_N) {
            const int j_end = (jj + BLOCK_N < N) ? jj + BLOCK_N : N;

            // Register blocking: 4 rows of A x 2 rows of B
            for (int i = ii; i + 3 < i_end; i += 4) {
                // Precompute base row pointers for A
                const float* a0_ptr = A + (i + 0) * lda;
                const float* a1_ptr = A + (i + 1) * lda;
                const float* a2_ptr = A + (i + 2) * lda;
                const float* a3_ptr = A + (i + 3) * lda;

                for (int j = jj; j + 1 < j_end; j += 2) {
                    // Precompute base row pointers for B
                    const float* b0_ptr = B + (j + 0) * ldb;
                    const float* b1_ptr = B + (j + 1) * ldb;

                    // ACCUMULATORS STAY IN YMM REGISTERS FOR THE ENTIRE K DIMENSION
                    __m256 v00 = _mm256_setzero_ps();
                    __m256 v01 = _mm256_setzero_ps();
                    __m256 v10 = _mm256_setzero_ps();
                    __m256 v11 = _mm256_setzero_ps();
                    __m256 v20 = _mm256_setzero_ps();
                    __m256 v21 = _mm256_setzero_ps();
                    __m256 v30 = _mm256_setzero_ps();
                    __m256 v31 = _mm256_setzero_ps();

                    int k = 0;
                    // Unroll along K by 16 floats (2 x AVX-256 steps)
                    for (; k + 15 < K; k += 16) {
                        // Software Prefetch next cache lines (~64 bytes ahead)
                        _mm_prefetch((const char*)(a0_ptr + k + 32), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b0_ptr + k + 32), _MM_HINT_T0);

                        // Step 1: k ... k+7
                        __m256 b0_0 = _mm256_loadu_ps(b0_ptr + k);
                        __m256 b1_0 = _mm256_loadu_ps(b1_ptr + k);

                        __m256 a0_0 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a1_0 = _mm256_loadu_ps(a1_ptr + k);
                        __m256 a2_0 = _mm256_loadu_ps(a2_ptr + k);
                        __m256 a3_0 = _mm256_loadu_ps(a3_ptr + k);

                        v00 = _mm256_fmadd_ps(a0_0, b0_0, v00);
                        v01 = _mm256_fmadd_ps(a0_0, b1_0, v01);
                        v10 = _mm256_fmadd_ps(a1_0, b0_0, v10);
                        v11 = _mm256_fmadd_ps(a1_0, b1_0, v11);
                        v20 = _mm256_fmadd_ps(a2_0, b0_0, v20);
                        v21 = _mm256_fmadd_ps(a2_0, b1_0, v21);
                        v30 = _mm256_fmadd_ps(a3_0, b0_0, v30);
                        v31 = _mm256_fmadd_ps(a3_0, b1_0, v31);

                        // Step 2: k+8 ... k+15
                        __m256 b0_1 = _mm256_loadu_ps(b0_ptr + k + 8);
                        __m256 b1_1 = _mm256_loadu_ps(b1_ptr + k + 8);

                        __m256 a0_1 = _mm256_loadu_ps(a0_ptr + k + 8);
                        __m256 a1_1 = _mm256_loadu_ps(a1_ptr + k + 8);
                        __m256 a2_1 = _mm256_loadu_ps(a2_ptr + k + 8);
                        __m256 a3_1 = _mm256_loadu_ps(a3_ptr + k + 8);

                        v00 = _mm256_fmadd_ps(a0_1, b0_1, v00);
                        v01 = _mm256_fmadd_ps(a0_1, b1_1, v01);
                        v10 = _mm256_fmadd_ps(a1_1, b0_1, v10);
                        v11 = _mm256_fmadd_ps(a1_1, b1_1, v11);
                        v20 = _mm256_fmadd_ps(a2_1, b0_1, v20);
                        v21 = _mm256_fmadd_ps(a2_1, b1_1, v21);
                        v30 = _mm256_fmadd_ps(a3_1, b0_1, v30);
                        v31 = _mm256_fmadd_ps(a3_1, b1_1, v31);
                    }

                    // 8-wide fallback for remaining K
                    for (; k + 7 < K; k += 8) {
                        __m256 b0 = _mm256_loadu_ps(b0_ptr + k);
                        __m256 b1 = _mm256_loadu_ps(b1_ptr + k);

                        __m256 a0 = _mm256_loadu_ps(a0_ptr + k);
                        __m256 a1 = _mm256_loadu_ps(a1_ptr + k);
                        __m256 a2 = _mm256_loadu_ps(a2_ptr + k);
                        __m256 a3 = _mm256_loadu_ps(a3_ptr + k);

                        v00 = _mm256_fmadd_ps(a0, b0, v00);
                        v01 = _mm256_fmadd_ps(a0, b1, v01);
                        v10 = _mm256_fmadd_ps(a1, b0, v10);
                        v11 = _mm256_fmadd_ps(a1, b1, v11);
                        v20 = _mm256_fmadd_ps(a2, b0, v20);
                        v21 = _mm256_fmadd_ps(a2, b1, v21);
                        v30 = _mm256_fmadd_ps(a3, b0, v30);
                        v31 = _mm256_fmadd_ps(a3, b1, v31);
                    }

                    // REDUCE HORIZONTALLY ONCE PER CELL AFTER K LOOP COMPLETES!
                    float c00 = hsum256(v00);
                    float c01 = hsum256(v01);
                    float c10 = hsum256(v10);
                    float c11 = hsum256(v11);
                    float c20 = hsum256(v20);
                    float c21 = hsum256(v21);
                    float c30 = hsum256(v30);
                    float c31 = hsum256(v31);

                    // Scalar tail for K % 8 != 0
                    for (; k < K; ++k) {
                        float b0_s = b0_ptr[k];
                        float b1_s = b1_ptr[k];

                        c00 += a0_ptr[k] * b0_s;
                        c01 += a0_ptr[k] * b1_s;
                        c10 += a1_ptr[k] * b0_s;
                        c11 += a1_ptr[k] * b1_s;
                        c20 += a2_ptr[k] * b0_s;
                        c21 += a2_ptr[k] * b1_s;
                        c30 += a3_ptr[k] * b0_s;
                        c31 += a3_ptr[k] * b1_s;
                    }

                    // Store results to Matrix C
                    C[(i + 0) * ldc + (j + 0)] = c00;
                    C[(i + 0) * ldc + (j + 1)] = c01;
                    C[(i + 1) * ldc + (j + 0)] = c10;
                    C[(i + 1) * ldc + (j + 1)] = c11;
                    C[(i + 2) * ldc + (j + 0)] = c20;
                    C[(i + 2) * ldc + (j + 1)] = c21;
                    C[(i + 3) * ldc + (j + 0)] = c30;
                    C[(i + 3) * ldc + (j + 1)] = c31;
                }

                // Cleanup for remaining J elements
                for (int j = (j_end & ~1); j < j_end; ++j) {
                    for (int r = i; r < i + 4; ++r) {
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k) {
                            sum += A[r * lda + k] * B[j * ldb + k];
                        }
                        C[r * ldc + j] = sum;
                    }
                }
            }

            // Cleanup for remaining I elements
            for (int i = (i_end & ~3); i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += A[i * lda + k] * B[j * ldb + k];
                    }
                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}