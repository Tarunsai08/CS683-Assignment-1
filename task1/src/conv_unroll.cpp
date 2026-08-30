// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // TODO(student): replace this placeholder with your unrolled implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // <<<------ Approach: Unrolled Kernel's and input's innerloop ------->>>
    for (int oy = 0; oy < H; ++oy) {
        // Unrolling 7 times (ox) since W is divisible by 8
        for (int ox = 0; ox < W; ox += 8) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx + 2 < K; kx += 3) {
                    // Unrolling 2 times (kx) since Kernel minimum size is 3 (works for multiples of 3)
                    acc0 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc0 += in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx + 1];
                    acc0 += in[(oy + ky) * in_stride + (ox + kx + 2)] * ker[ky * K + kx + 2];

                    acc1 += in[(oy + ky) * in_stride + (ox + 1 + kx)] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + (ox + 1 + kx + 1)] * ker[ky * K + kx + 1];
                    acc1 += in[(oy + ky) * in_stride + (ox + 1 + kx + 2)] * ker[ky * K + kx + 2];

                    acc2 += in[(oy + ky) * in_stride + (ox + 2 + kx)] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + (ox + 2 + kx + 1)] * ker[ky * K + kx + 1];
                    acc2 += in[(oy + ky) * in_stride + (ox + 2 + kx + 2)] * ker[ky * K + kx + 2];

                    acc3 += in[(oy + ky) * in_stride + (ox + 3 + kx)] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + (ox + 3 + kx + 1)] * ker[ky * K + kx + 1];
                    acc3 += in[(oy + ky) * in_stride + (ox + 3 + kx + 2)] * ker[ky * K + kx + 2];

                    acc4 += in[(oy + ky) * in_stride + (ox + 4 + kx)] * ker[ky * K + kx];
                    acc4 += in[(oy + ky) * in_stride + (ox + 4 + kx + 1)] * ker[ky * K + kx + 1];
                    acc4 += in[(oy + ky) * in_stride + (ox + 4 + kx + 2)] * ker[ky * K + kx + 2];

                    acc5 += in[(oy + ky) * in_stride + (ox + 5 + kx)] * ker[ky * K + kx];
                    acc5 += in[(oy + ky) * in_stride + (ox + 5 + kx + 1)] * ker[ky * K + kx + 1];
                    acc5 += in[(oy + ky) * in_stride + (ox + 5 + kx + 2)] * ker[ky * K + kx + 2];

                    acc6 += in[(oy + ky) * in_stride + (ox + 6 + kx)] * ker[ky * K + kx];
                    acc6 += in[(oy + ky) * in_stride + (ox + 6 + kx + 1)] * ker[ky * K + kx + 1];
                    acc6 += in[(oy + ky) * in_stride + (ox + 6 + kx + 2)] * ker[ky * K + kx + 2];

                    acc7 += in[(oy + ky) * in_stride + (ox + 7 + kx)] * ker[ky * K + kx];
                    acc7 += in[(oy + ky) * in_stride + (ox + 7 + kx + 1)] * ker[ky * K + kx + 1];
                    acc7 += in[(oy + ky) * in_stride + (ox + 7 + kx + 2)] * ker[ky * K + kx + 2];
                }
                for (int kx = (K / 3) * 3; kx < K; ++kx) {
                    // For any left out kernel indices, will run atmost 2 times
                    acc0 += in[(oy + ky) * in_stride + ox + kx] * ker[ky * K + kx];
                    acc1 += in[(oy + ky) * in_stride + ox + 1 + kx] * ker[ky * K + kx];
                    acc2 += in[(oy + ky) * in_stride + ox + 2 + kx] * ker[ky * K + kx];
                    acc3 += in[(oy + ky) * in_stride + ox + 3 + kx] * ker[ky * K + kx];
                    acc4 += in[(oy + ky) * in_stride + ox + 4 + kx] * ker[ky * K + kx];
                    acc5 += in[(oy + ky) * in_stride + ox + 5 + kx] * ker[ky * K + kx];
                    acc6 += in[(oy + ky) * in_stride + ox + 6 + kx] * ker[ky * K + kx];
                    acc7 += in[(oy + ky) * in_stride + ox + 7 + kx] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox]     = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
            out[oy * W + ox + 4] = acc4;
            out[oy * W + ox + 5] = acc5;
            out[oy * W + ox + 6] = acc6;
            out[oy * W + ox + 7] = acc7;
        }
    }
}
