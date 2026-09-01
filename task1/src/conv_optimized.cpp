// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
using namespace std;

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int TILE = 1;

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int ii = 0; ii < H; ii += TILE) {
        for (int jj = 0; jj < W; jj += TILE) {

            const int i_end = min(ii + TILE, H);
            const int j_end = min(jj + TILE, W);

            for (int oy = ii; oy < i_end; ++oy) {
                for (int ox = jj; ox < j_end; ox += 8) {

                    __m256 acc = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky) {

                        const float* in_row =
                            &in[(oy + ky) * in_stride + ox];

                        const float* ker_row =
                            &ker[ky * K];

                        int kx = 0;

                        for (; kx + 3 < K; kx += 4) {
                            __m256 k0 = _mm256_set1_ps(ker_row[kx]);
                            __m256 k1 = _mm256_set1_ps(ker_row[kx + 1]);
                            __m256 k2 = _mm256_set1_ps(ker_row[kx + 2]);
                            __m256 k3 = _mm256_set1_ps(ker_row[kx + 3]);

                            __m256 x0 = _mm256_loadu_ps(in_row + kx);
                            __m256 x1 = _mm256_loadu_ps(in_row + kx + 1);
                            __m256 x2 = _mm256_loadu_ps(in_row + kx + 2);
                            __m256 x3 = _mm256_loadu_ps(in_row + kx + 3);

                            acc = _mm256_fmadd_ps(x0, k0, acc);
                            acc = _mm256_fmadd_ps(x1, k1, acc);
                            acc = _mm256_fmadd_ps(x2, k2, acc);
                            acc = _mm256_fmadd_ps(x3, k3, acc);
                        }

                        for (; kx < K; ++kx) {
                            __m256 input = _mm256_loadu_ps(in_row + kx);
                            __m256 kernel = _mm256_set1_ps(ker_row[kx]);
                            acc = _mm256_fmadd_ps(input, kernel, acc);
                        }
                    }

                    _mm256_storeu_ps(
                        &out[oy * W + ox],
                        acc
                    );
                }
            }
        }
    }
}