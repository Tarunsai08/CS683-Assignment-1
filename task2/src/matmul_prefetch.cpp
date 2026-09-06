#include <immintrin.h>
#include <algorithm>
#include <cstddef>

#include "matmul.h"

static constexpr int JB = 4;  // register-block width: output columns computed together

// Target working-set size (A-panel + B-panel) per tile, in bytes. Sized for this
// for C writes and other cache pressure. Tune this constant if you move to another CPU.

static constexpr std::size_t kTargetBytes = 600 * 1024;

// Picks BM (rows of A per tile) and BN (rows of B per tile) so that
// (BM + BN) * K * 4 bytes stays close to kTargetBytes, regardless of K.
// B gets a larger share since it is reused across every row of the A-tile.

static inline void compute_tile_sizes(int K, int& BM, int& BN) {
    const std::size_t bytes_per_row = static_cast<std::size_t>(K) * sizeof(float);
    std::size_t max_rows = kTargetBytes / bytes_per_row;
    if (max_rows < 8) max_rows = 8;  // floor so tiny-K cases still get a sane tile

    BN = static_cast<int>((max_rows * 4) / 5);  // ~80% of the budget to B
    BM = static_cast<int>(max_rows - static_cast<std::size_t>(BN));  // remainder to A

    BN -= BN % JB;              // keep BN a multiple of the register-block width
    if (BN < JB) BN = JB;
    if (BM < 1) BM = 1;
}

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    __m128 su = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, su);
    su = _mm_add_ss(su, sh);
    return _mm_cvtss_f32(su);
}

// Computes C[i, j..j+3] for one row i, given 4 rows of B, prefetching one row of
// B ahead (from the NEXT group of 4, if any) to hide latency within the tile.
static inline void block4_simd_prefetch(const float* a_row,
                                         const float* b0, const float* b1,
                                         const float* b2, const float* b3,
                                         const float* b_prefetch,  // row to prefetch, or nullptr
                                         int K, float* out /*[4]*/) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8) {
        __m256 va = _mm256_loadu_ps(a_row + p);
        acc0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), acc0);
        acc1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), acc1);
        acc2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p), acc2);
        acc3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p), acc3);
        if (b_prefetch) {
            _mm_prefetch(reinterpret_cast<const char*>(b_prefetch + p), _MM_HINT_T0);
        }
    }

    float r0 = hsum256(acc0), r1 = hsum256(acc1);
    float r2 = hsum256(acc2), r3 = hsum256(acc3);

    for (; p < K; ++p) {
        float av = a_row[p];
        r0 += av * b0[p];
        r1 += av * b1[p];
        r2 += av * b2[p];
        r3 += av * b3[p];
    }

    out[0] = r0; out[1] = r1; out[2] = r2; out[3] = r3;
}

// Scalar-ish (still SIMD over K) fallback for one output column, for the trailing
// columns when a tile's width isn't a multiple of JB.
static inline float dot_simd(const float* a, const float* b, int K) {
    __m256 acc = _mm256_setzero_ps();
    int p = 0;
    for (; p + 8 <= K; p += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
    }
    float r = hsum256(acc);
    for (; p < K; ++p) r += a[p] * b[p];
    return r;
}

void matmul_prefetch(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    int BM, BN;
    compute_tile_sizes(K, BM, BN);

    for (int j0 = 0; j0 < N; j0 += BN) {           // B-panel outermost: reused across all i0 blocks
        const int jEnd = std::min(j0 + BN, N);

        for (int i0 = 0; i0 < M; i0 += BM) {
            const int iEnd = std::min(i0 + BM, M);

            for (int i = i0; i < iEnd; ++i) {
                const float* a_row = A + static_cast<std::size_t>(i) * lda;
                if (i + 1 < iEnd) {
                    _mm_prefetch(reinterpret_cast<const char*>(a_row + lda), _MM_HINT_T0);
                }

                float* c_row = C + static_cast<std::size_t>(i) * ldc;

                int j = j0;
                for (; j + JB <= jEnd; j += JB) {
                    const float* b0 = B + static_cast<std::size_t>(j + 0) * ldb;
                    const float* b1 = B + static_cast<std::size_t>(j + 1) * ldb;
                    const float* b2 = B + static_cast<std::size_t>(j + 2) * ldb;
                    const float* b3 = B + static_cast<std::size_t>(j + 3) * ldb;

                    const float* b_next_group =
                        (j + JB + JB <= jEnd) ? (B + static_cast<std::size_t>(j + JB) * ldb)
                                              : nullptr;

                    float out[4];
                    block4_simd_prefetch(a_row, b0, b1, b2, b3, b_next_group, K, out);
                    c_row[j + 0] = out[0];
                    c_row[j + 1] = out[1];
                    c_row[j + 2] = out[2];
                    c_row[j + 3] = out[3];
                }

                for (; j < jEnd; ++j) {
                    const float* b_row = B + static_cast<std::size_t>(j) * ldb;
                    c_row[j] = dot_simd(a_row, b_row, K);
                }
            }
        }
    }
}