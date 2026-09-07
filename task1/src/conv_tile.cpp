#include "convolution.h"
#include <cstring>

#define min(a,b) a < b ? a : b

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Row-strip height: sized so the strip's input+output rows stay
    // resident in L2 across all K*K tap passes. Sweep 16/32/64/128.
    const int TILE_ROWS = 64;

    std::memset(out, 0, sizeof(float) * (size_t)H * W);

    for (int ii = 0; ii < H; ii += TILE_ROWS) {
        int i_end = min(ii + TILE_ROWS, H);

        // Tap-sweep confined to this row strip: for each kernel tap,
        // sweep the whole strip with a contiguous, vectorizable add.
        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {
                const float kval = ker[ky * K + kx];

                for (int oy = ii; oy < i_end; oy++) {
                    const float* in_row  = in + (oy + ky) * in_stride + kx;
                    float*       out_row = out + oy * W;

                    for (int ox = 0; ox < W; ox++) {
                        out_row[ox] += kval * in_row[ox];
                    }
                }
            }
        }
    }
}