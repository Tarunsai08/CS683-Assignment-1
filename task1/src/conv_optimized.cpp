#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    constexpr int TILE = 2048;

    for (int ii = 0; ii < H; ii += TILE) {
        const int i_end = (ii + TILE < H) ? ii + TILE : H;

        for (int jj = 0; jj < W; jj += TILE) {
            const int j_end = (jj + TILE < W) ? jj + TILE : W;

            for (int oy = ii; oy < i_end; ++oy) {
                float* out_row = out + oy * W;

                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row =
                        in + (oy + ky) * in_stride;

                    const float* ker_row =
                        ker + ky * K;

                    for (int kx = 0; kx < K; ++kx) {
                        const float w = ker_row[kx];
                        const __m256 kernel = _mm256_set1_ps(w);

                        int ox = jj;

                        for (; ox + 32 <= j_end; ox += 32) {
                            const float* in_ptr = in_row + ox + kx;

                            __m256 v0 = _mm256_loadu_ps(in_ptr);
                            __m256 v1 = _mm256_loadu_ps(in_ptr + 8);
                            __m256 v2 = _mm256_loadu_ps(in_ptr + 16);
                            __m256 v3 = _mm256_loadu_ps(in_ptr + 24);

                            if (ky == 0 && kx == 0) {
                                _mm256_storeu_ps(
                                    out_row + ox,
                                    _mm256_mul_ps(v0, kernel));

                                _mm256_storeu_ps(
                                    out_row + ox + 8,
                                    _mm256_mul_ps(v1, kernel));

                                _mm256_storeu_ps(
                                    out_row + ox + 16,
                                    _mm256_mul_ps(v2, kernel));

                                _mm256_storeu_ps(
                                    out_row + ox + 24,
                                    _mm256_mul_ps(v3, kernel));
                            } else {
                                __m256 o0 =
                                    _mm256_loadu_ps(out_row + ox);
                                __m256 o1 =
                                    _mm256_loadu_ps(out_row + ox + 8);
                                __m256 o2 =
                                    _mm256_loadu_ps(out_row + ox + 16);
                                __m256 o3 =
                                    _mm256_loadu_ps(out_row + ox + 24);

                                o0 = _mm256_fmadd_ps(v0, kernel, o0);
                                o1 = _mm256_fmadd_ps(v1, kernel, o1);
                                o2 = _mm256_fmadd_ps(v2, kernel, o2);
                                o3 = _mm256_fmadd_ps(v3, kernel, o3);

                                _mm256_storeu_ps(out_row + ox, o0);
                                _mm256_storeu_ps(out_row + ox + 8, o1);
                                _mm256_storeu_ps(out_row + ox + 16, o2);
                                _mm256_storeu_ps(out_row + ox + 24, o3);
                            }
                        }

                        for (; ox + 8 <= j_end; ox += 8) {
                            const float* in_ptr =
                                in_row + ox + kx;

                            __m256 input =
                                _mm256_loadu_ps(in_ptr);

                            if (ky == 0 && kx == 0) {
                                _mm256_storeu_ps(
                                    out_row + ox,
                                    _mm256_mul_ps(input, kernel));
                            } else {
                                __m256 output =
                                    _mm256_loadu_ps(out_row + ox);

                                output =
                                    _mm256_fmadd_ps(
                                        input, kernel, output);

                                _mm256_storeu_ps(
                                    out_row + ox, output);
                            }
                        }

                        for (; ox < j_end; ++ox) {
                            float value =
                                in_row[ox + kx] * w;

                            if (ky == 0 && kx == 0)
                                out_row[ox] = value;
                            else
                                out_row[ox] += value;
                        }
                    }
                }
            }
        }
    }
}