// // conv_tile.cpp  STAGE 3: CACHE TILING

// // conv_tile version 1.0 

// #include "convolution.h"
// #include "algorithm"

// using namespace std;


// void conv_tile(const float* in, float* out, const float* ker,
//                int H, int W, int K) {
//     const int TILE = 32;
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;

//     for (int ii = 0; ii < H; ii += TILE) {
//         for (int jj = 0; jj < W; jj += TILE) {

//             int i_end = min(ii+TILE,H); 
//             int j_end = min(jj+TILE,W); 

//             for (int oy = ii; oy < i_end; oy++) {
//                 for (int ox = jj; ox < j_end; ox++) {

//                     float acc = 0.0f;
//                     for (int ky = 0; ky < K; ky++) {
//                         for (int kx = 0; kx < K; kx++) {
//                             acc += in[(oy + ky) * in_stride + (ox + kx)] *
//                                    ker[ky * K + kx];
//                         }
//                     }
//                     out[oy * W + ox] = acc;
//                 }
//             }
//         }
//     }
// }

// conv_tile.cpp  STAGE 3: CACHE TILING (row-strip + tap-sweep)

#include "convolution.h"
#include <cstring>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // Row-strip height: pick so (strip rows) x (row bytes, in+out) fits ~L2.
    // Tune this — start around 64 and sweep 16/32/64/128 to see what wins.
    
    const int TILE_ROWS = 64;

    std::memset(out, 0, sizeof(float) * (size_t)H * W);

    for (int ii = 0; ii < H; ii += TILE_ROWS) {
        int i_end = (ii + TILE_ROWS < H) ? (ii + TILE_ROWS) : H;

        // Same tap-sweep as reorder, but confined to this row strip so the
        // strip's input rows + output rows stay hot in cache across all K*K taps.
        
        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {
                const float kval = ker[ky * K + kx];
                for (int oy = ii; oy < i_end; oy++) {
                    const float* in_row  = in + (oy + ky) * in_stride + kx;
                    float* out_row = out + oy * W;
                    for (int ox = 0; ox < W; ox++) {
                        out_row[ox] += kval * in_row[ox];
                    }
                }
            }
        }
    }
} 