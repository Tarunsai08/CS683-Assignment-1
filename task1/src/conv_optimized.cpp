#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Test these experimentally:
    // 32, 64, 128, 256
    constexpr int TILE = 2048;

    for (int ii = 0; ii < H; ii += TILE) {
        const int i_end = (ii + TILE < H) ? ii + TILE : H;

        for (int jj = 0; jj < W; jj += TILE) {
            const int j_end = (jj + TILE < W) ? jj + TILE : W;

            for (int oy = ii; oy < i_end; ++oy) {
                const float* in_base = in + oy * in_stride;
                float* out_row = out + oy * W;

                int ox = jj;

                // SIMD: 8 output pixels at a time.
                for (; ox + 8 <= j_end; ox += 8) {
                    __m256 acc = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row =
                            in_base + ky * in_stride + ox;

                        const float* ker_row =
                            ker + ky * K;

                        for (int kx = 0; kx < K; ++kx) {
                            const __m256 input =
                                _mm256_loadu_ps(in_row + kx);

                            const __m256 kernel =
                                _mm256_set1_ps(ker_row[kx]);

                            acc = _mm256_fmadd_ps(input, kernel, acc);
                        }
                    }

                    _mm256_storeu_ps(out_row + ox, acc);
                }

                // Scalar tail inside the tile.
                for (; ox < j_end; ++ox) {
                    float acc = 0.0f;

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row =
                            in_base + ky * in_stride + ox;

                        const float* ker_row =
                            ker + ky * K;

                        for (int kx = 0; kx < K; ++kx) {
                            acc += in_row[kx] * ker_row[kx];
                        }
                    }

                    out_row[ox] = acc;
                }
            }
        }
    }
}