// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    constexpr int TILE_H = 32;
    constexpr int TILE_W = 256;

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int ii = 0; ii < H; ii += TILE_H) {
        const int i_end = (ii + TILE_H < H) ? ii + TILE_H : H;

        for (int jj = 0; jj < W; jj += TILE_W) {
            const int j_end = (jj + TILE_W < W) ? jj + TILE_W : W;

            for (int oy = ii; oy < i_end; ++oy) {
                float* out_row = out + oy * W;
                int ox = jj;

                for (; ox + 32 <= j_end; ox += 32) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_ptr = in + (oy + ky) * in_stride + ox;
                        const float* ker_row = ker + ky * K;

                        for (int kx = 0; kx < K; ++kx) {
                            __m256 kernel = _mm256_set1_ps(ker_row[kx]);

                            __m256 v0 = _mm256_loadu_ps(in_ptr + kx);
                            __m256 v1 = _mm256_loadu_ps(in_ptr + kx + 8);
                            __m256 v2 = _mm256_loadu_ps(in_ptr + kx + 16);
                            __m256 v3 = _mm256_loadu_ps(in_ptr + kx + 24);

                            acc0 = _mm256_fmadd_ps(v0, kernel, acc0);
                            acc1 = _mm256_fmadd_ps(v1, kernel, acc1);
                            acc2 = _mm256_fmadd_ps(v2, kernel, acc2);
                            acc3 = _mm256_fmadd_ps(v3, kernel, acc3);
                        }
                    }

                    _mm256_storeu_ps(out_row + ox,      acc0);
                    _mm256_storeu_ps(out_row + ox + 8,  acc1);
                    _mm256_storeu_ps(out_row + ox + 16, acc2);
                    _mm256_storeu_ps(out_row + ox + 24, acc3);
                }

                for (; ox + 8 <= j_end; ox += 8) {
                    __m256 acc = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_ptr = in + (oy + ky) * in_stride + ox;
                        const float* ker_row = ker + ky * K;

                        for (int kx = 0; kx < K; ++kx) {
                            __m256 kernel = _mm256_set1_ps(ker_row[kx]);
                            __m256 v = _mm256_loadu_ps(in_ptr + kx);
                            acc = _mm256_fmadd_ps(v, kernel, acc);
                        }
                    }
                    _mm256_storeu_ps(out_row + ox, acc);
                }

                for (; ox < j_end; ++ox) {
                    float val = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_ptr = in + (oy + ky) * in_stride + ox;
                        const float* ker_row = ker + ky * K;
                        for (int kx = 0; kx < K; ++kx) {
                            val += in_ptr[kx] * ker_row[kx];
                        }
                    }
                    out_row[ox] = val;
                }
            }
        }
    }
}