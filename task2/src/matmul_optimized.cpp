#include <immintrin.h>
#include "matmul.h"

static inline float hsum256(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);

    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum);
}

static inline void kernel_4x2(
    const float* A,
    const float* B,
    float* C,
    int i,
    int j,
    int K,
    int lda,
    int ldb,
    int ldc) {

    float s00 = 0.0f;
    float s01 = 0.0f;
    float s10 = 0.0f;
    float s11 = 0.0f;
    float s20 = 0.0f;
    float s21 = 0.0f;
    float s30 = 0.0f;
    float s31 = 0.0f;

    constexpr int BK = 256;

    for (int kk = 0; kk < K; kk += BK) {
        const int kend = (kk + BK < K) ? kk + BK : K;

        __m256 acc00 = _mm256_setzero_ps();
        __m256 acc01 = _mm256_setzero_ps();
        __m256 acc10 = _mm256_setzero_ps();
        __m256 acc11 = _mm256_setzero_ps();
        __m256 acc20 = _mm256_setzero_ps();
        __m256 acc21 = _mm256_setzero_ps();
        __m256 acc30 = _mm256_setzero_ps();
        __m256 acc31 = _mm256_setzero_ps();

        int k = kk;

        for (; k + 7 < kend; k += 8) {
            __m256 b0 =
                _mm256_loadu_ps(B + j * ldb + k);
            __m256 b1 =
                _mm256_loadu_ps(B + (j + 1) * ldb + k);

            __m256 a0 =
                _mm256_loadu_ps(A + i * lda + k);
            __m256 a1 =
                _mm256_loadu_ps(A + (i + 1) * lda + k);
            __m256 a2 =
                _mm256_loadu_ps(A + (i + 2) * lda + k);
            __m256 a3 =
                _mm256_loadu_ps(A + (i + 3) * lda + k);

            acc00 = _mm256_fmadd_ps(a0, b0, acc00);
            acc01 = _mm256_fmadd_ps(a0, b1, acc01);

            acc10 = _mm256_fmadd_ps(a1, b0, acc10);
            acc11 = _mm256_fmadd_ps(a1, b1, acc11);

            acc20 = _mm256_fmadd_ps(a2, b0, acc20);
            acc21 = _mm256_fmadd_ps(a2, b1, acc21);

            acc30 = _mm256_fmadd_ps(a3, b0, acc30);
            acc31 = _mm256_fmadd_ps(a3, b1, acc31);
        }

        s00 += hsum256(acc00);
        s01 += hsum256(acc01);
        s10 += hsum256(acc10);
        s11 += hsum256(acc11);
        s20 += hsum256(acc20);
        s21 += hsum256(acc21);
        s30 += hsum256(acc30);
        s31 += hsum256(acc31);

        for (; k < kend; ++k) {
            const float b0 = B[j * ldb + k];
            const float b1 = B[(j + 1) * ldb + k];

            const float a0 = A[i * lda + k];
            const float a1 = A[(i + 1) * lda + k];
            const float a2 = A[(i + 2) * lda + k];
            const float a3 = A[(i + 3) * lda + k];

            s00 += a0 * b0;
            s01 += a0 * b1;

            s10 += a1 * b0;
            s11 += a1 * b1;

            s20 += a2 * b0;
            s21 += a2 * b1;

            s30 += a3 * b0;
            s31 += a3 * b1;
        }
    }

    C[i * ldc + j] = s00;
    C[i * ldc + j + 1] = s01;

    C[(i + 1) * ldc + j] = s10;
    C[(i + 1) * ldc + j + 1] = s11;

    C[(i + 2) * ldc + j] = s20;
    C[(i + 2) * ldc + j + 1] = s21;

    C[(i + 3) * ldc + j] = s30;
    C[(i + 3) * ldc + j + 1] = s31;
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      int lda, int ldb, int ldc) {

    constexpr int BM = 128;
    constexpr int BN = 32;

    for (int ii = 0; ii < M; ii += BM) {
        const int i_end = (ii + BM < M) ? ii + BM : M;

        for (int jj = 0; jj < N; jj += BN) {
            const int j_end = (jj + BN < N) ? jj + BN : N;

            int i = ii;

            for (; i + 3 < i_end; i += 4) {
                int j = jj;

                for (; j + 1 < j_end; j += 2) {
                    kernel_4x2(
                        A, B, C,
                        i, j,
                        K,
                        lda, ldb, ldc);
                }

                for (; j < j_end; ++j) {
                    for (int r = i; r < i + 4; ++r) {
                        float sum = 0.0f;

                        for (int k = 0; k < K; ++k) {
                            sum += A[r * lda + k] *
                                   B[j * ldb + k];
                        }

                        C[r * ldc + j] = sum;
                    }
                }
            }

            for (; i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    float sum = 0.0f;

                    for (int k = 0; k < K; ++k) {
                        sum += A[i * lda + k] *
                               B[j * ldb + k];
                    }

                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}