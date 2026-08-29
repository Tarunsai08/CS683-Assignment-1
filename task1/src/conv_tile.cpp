// conv_tile.cpp  STAGE 3: CACHE TILING

// conv_tile version 1.0 

#include "convolution.h"
#include "algorithm"

using namespace std;


void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int TILE = 32;
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int ii = 0; ii < H; ii += TILE) {
        for (int jj = 0; jj < W; jj += TILE) {

            int i_end = min(ii+TILE,H); // (ii + TILE) < H ? (ii + TILE) : H;
            int j_end = (jj + TILE) < W ? (jj + TILE) : W;

            for (int oy = ii; oy < i_end; oy++) {
                for (int ox = jj; ox < j_end; ox++) {

                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ky++) {
                        for (int kx = 0; kx < K; kx++) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] *
                                   ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}
