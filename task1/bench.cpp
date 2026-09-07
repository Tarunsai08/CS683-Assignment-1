#include <cstdio>
#include <cstdlib>
#include "convolution.h"

int main() {
    constexpr int H = 2048;
    constexpr int W = 2048;
    constexpr int K = 3;

    constexpr int p = K / 2;
    constexpr int stride = W + 2 * p;

    const size_t input_size =
        static_cast<size_t>(H + 2 * p) * stride;

    const size_t output_size =
        static_cast<size_t>(H) * W;

    const size_t kernel_size =
        static_cast<size_t>(K) * K;

    float* in =
        static_cast<float*>(
            std::aligned_alloc(32, input_size * sizeof(float)));

    float* out =
        static_cast<float*>(
            std::aligned_alloc(32, output_size * sizeof(float)));

    float* ker =
        static_cast<float*>(
            std::aligned_alloc(32, kernel_size * sizeof(float)));

    if (!in || !out || !ker) {
        std::fprintf(stderr, "allocation failed\n");
        return 1;
    }

    // Deterministic data.
    for (size_t i = 0; i < input_size; ++i)
        in[i] = 1.0f;

    for (size_t i = 0; i < output_size; ++i)
        out[i] = 0.0f;

    for (size_t i = 0; i < kernel_size; ++i)
        ker[i] = 1.0f;

    // Warmup.
    for (int i = 0; i < 10; ++i)
        conv_optimized(in, out, ker, H, W, K);

    // Lots of repetitions so perf measures almost entirely
    // conv_optimized().
    for (int i = 0; i < 100; ++i)
        conv_optimized(in, out, ker, H, W, K);

    // Prevent the result from being completely irrelevant.
    std::printf("out[0] = %f\n", out[0]);

    std::free(in);
    std::free(out);
    std::free(ker);

    return 0;
}