// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // TODO(student): replace this placeholder with your reordered implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;

        for (int ky = 0; ky < K; ++ky) {
            const float* in_row = in + (oy + ky) * in_stride;
            const float* krow = ker + ky * K;

            for (int kx = 0; kx < K; ++kx) {
                const float w = krow[kx];
                const float* in_ptr = in_row + kx;

                if (ky == 0 && kx == 0) {
                    float* out_ptr = out_row;

                    for (int ox = 0; ox < W; ++ox) {
                        *out_ptr = *in_ptr * w;
                        ++out_ptr;
                        ++in_ptr;
                    }
                } else {
                    float* out_ptr = out_row;

                    for (int ox = 0; ox < W; ++ox) {
                        *out_ptr += *in_ptr * w;
                        ++out_ptr;
                        ++in_ptr;
                    }
                }
            }
        }
    }
}