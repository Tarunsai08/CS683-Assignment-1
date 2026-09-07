#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CS683 PA-1 TASK 1 DATA COLLECTION
#
# Run from the task1 directory containing:
#   src/conv_naive.cpp
#   src/conv_reorder.cpp
#   src/conv_unroll.cpp
#   src/conv_tile.cpp
#   src/conv_simd.cpp
#   src/conv_optimized.cpp
#   src/main.cpp
#   bin/conv (built by make)
#
# This script:
#   1. Measures timing/speedup across matrix sizes.
#   2. Sweeps tile sizes for Task 1B.
#   3. Measures process-level instructions and L1-D MPKI using perf.
#   4. Creates a temporary 128-bit SIMD version for Task 1C.
#   5. Leaves your original source files restored.
#
# IMPORTANT:
# The provided grading harness times a stage correctly, but perf
# around ./bin/conv also includes the harness/reference work.
# Therefore the perf MPKI/instruction values collected here are
# PROCESS-LEVEL measurements. Use them for the report only if
# your instructor accepts harness-level perf measurements.
# For stage-only counters, see the notes printed at the end.
# ============================================================

OUT="task1_results"
mkdir -p "$OUT"

SIZES=(256 512 1024 2048)
KERNELS=(3 5)
TILES=(8 16 32 48 64 80 96 128 256 512 1024 2048)

mkdir -p "$OUT/raw"

BACKUP_TILE="$OUT/.conv_tile.cpp.bak"
BACKUP_SIMD="$OUT/.conv_simd.cpp.bak"
cp src/conv_tile.cpp "$BACKUP_TILE"
cp src/conv_simd.cpp "$BACKUP_SIMD"

restore() {
    cp "$BACKUP_TILE" src/conv_tile.cpp
    cp "$BACKUP_SIMD" src/conv_simd.cpp
    make -s >/dev/null 2>&1 || true
}
trap restore EXIT INT TERM

build() {
    make clean >/dev/null 2>&1 || true
    make >/dev/null
}

get_time() {
    local f="$1" stage="$2"
    if [[ "$stage" == "naive" ]]; then
        awk '$1=="naive" && $2=="(ref)" {print $4; exit}' "$f"
    else
        awk -v s="$stage" '$1==s {print $3; exit}' "$f"
    fi
}

get_gflops() {
    local f="$1" stage="$2"
    if [[ "$stage" == "naive" ]]; then
        awk '$1=="naive" && $2=="(ref)" {print $6; exit}' "$f"
    else
        awk -v s="$stage" '$1==s {print $5; exit}' "$f"
    fi
}

perf_value() {
    local f="$1" event="$2"
    awk -v e="$event" '
        $0 ~ e {
            for (i=1; i<=NF; i++) {
                x=$i
                gsub(",", "", x)
                if (x ~ /^[0-9]+([.][0-9]+)?$/) {
                    print x
                    exit
                }
            }
        }
    ' "$f"
}

# ------------------------------------------------------------
# Timing/speedup data for all stages and sizes.
# ------------------------------------------------------------
echo "size,K,naive_ms,reorder_ms,reorder_speedup,unroll_ms,unroll_speedup,tile_ms,tile_speedup,simd_ms,simd_speedup,optimized_ms,optimized_speedup" \
    > "$OUT/task1_stages.csv"

for N in "${SIZES[@]}"; do
    for K in "${KERNELS[@]}"; do
        raw="$OUT/raw/all_${N}_${K}.txt"
        ./bin/conv all "$N" "$N" "$K" > "$raw"

        naive=$(get_time "$raw" naive)
        reorder=$(get_time "$raw" reorder)
        unroll=$(get_time "$raw" unroll)
        tile=$(get_time "$raw" tile)
        simd=$(get_time "$raw" simd)
        optimized=$(get_time "$raw" optimized)

        rs=$(awk -v a="$naive" -v b="$reorder" 'BEGIN{printf "%.6f",a/b}')
        us=$(awk -v a="$naive" -v b="$unroll" 'BEGIN{printf "%.6f",a/b}')
        ts=$(awk -v a="$naive" -v b="$tile" 'BEGIN{printf "%.6f",a/b}')
        ss=$(awk -v a="$naive" -v b="$simd" 'BEGIN{printf "%.6f",a/b}')
        os=$(awk -v a="$naive" -v b="$optimized" 'BEGIN{printf "%.6f",a/b}')

        echo "$N,$K,$naive,$reorder,$rs,$unroll,$us,$tile,$ts,$simd,$ss,$optimized,$os" >> "$OUT/task1_stages.csv"
        echo "STAGES N=$N K=$K: naive=$naive reorder=$rs unroll=$us tile=$ts simd=$ss optimized=$os"
    done
done

# ------------------------------------------------------------
# Task 1B: tile sweep.
# Note: conv_tile.cpp has TILE as a source constant.
# ------------------------------------------------------------
echo "size,tile,naive_ms,tile_ms,tile_speedup,process_instructions,process_L1_loads,process_L1_misses,process_L1_MPKI" \
    > "$OUT/task1B.csv"

for N in "${SIZES[@]}"; do
    for T in "${TILES[@]}"; do
        sed -i -E "s/(const int TILE = )[0-9]+;/\1${T};/" src/conv_tile.cpp
        build

        timing="$OUT/raw/tile_${N}_${T}.txt"
        perfraw="$OUT/raw/perf_tile_${N}_${T}.txt"

        ./bin/conv tile "$N" "$N" 3 > "$timing"

        naive_ms=$(get_time "$timing" naive)
        tile_ms=$(get_time "$timing" tile)
        speed=$(awk -v a="$naive_ms" -v b="$tile_ms" 'BEGIN{printf "%.6f",a/b}')

        perf stat -r 3 -e instructions,L1-dcache-loads,L1-dcache-load-misses \
            ./bin/conv tile "$N" "$N" 3 >/dev/null 2>"$perfraw" || true

        ins=$(perf_value "$perfraw" "instructions")
        l1loads=$(perf_value "$perfraw" "L1-dcache-loads")
        l1miss=$(perf_value "$perfraw" "L1-dcache-load-misses")
        mpki=$(awk -v m="${l1miss:-0}" -v i="${ins:-0}" \
            'BEGIN{if(i>0) printf "%.6f",1000*m/i; else print "NA"}')

        echo "$N,$T,$naive_ms,$tile_ms,$speed,${ins:-NA},${l1loads:-NA},${l1miss:-NA},$mpki" \
            >> "$OUT/task1B.csv"

        echo "TILE N=$N T=$T speedup=$speed MPKI=$mpki"
    done
done

# ------------------------------------------------------------
# Task 1C: current 256-bit SIMD.
# ------------------------------------------------------------
echo "size,width_bits,time_ms,speedup,instructions,L1_loads,L1_misses,L1_MPKI" \
    > "$OUT/task1C.csv"

# Restore original SIMD first.
cp "$BACKUP_SIMD" src/conv_simd.cpp
build

for N in "${SIZES[@]}"; do
    timing="$OUT/raw/simd256_${N}.txt"
    perfraw="$OUT/raw/perf_simd256_${N}.txt"

    ./bin/conv simd "$N" "$N" 3 > "$timing"
    naive_ms=$(get_time "$timing" naive)
    simd_ms=$(get_time "$timing" simd)
    speed=$(awk -v a="$naive_ms" -v b="$simd_ms" 'BEGIN{printf "%.6f",a/b}')

    perf stat -r 3 -e instructions,L1-dcache-loads,L1-dcache-load-misses \
        ./bin/conv simd "$N" "$N" 3 >/dev/null 2>"$perfraw" || true

    ins=$(perf_value "$perfraw" "instructions")
    l1loads=$(perf_value "$perfraw" "L1-dcache-loads")
    l1miss=$(perf_value "$perfraw" "L1-dcache-load-misses")
    mpki=$(awk -v m="${l1miss:-0}" -v i="${ins:-0}" \
        'BEGIN{if(i>0) printf "%.6f",1000*m/i; else print "NA"}')

    echo "$N,256,$simd_ms,$speed,${ins:-NA},${l1loads:-NA},${l1miss:-NA},$mpki" >> "$OUT/task1C.csv"
done

# ------------------------------------------------------------
# Create a temporary 128-bit implementation.
# Same algorithm/order, 4 floats per vector.
# ------------------------------------------------------------
cat > src/conv_simd.cpp <<'EOF'
#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 4) {
            __m128 acc = _mm_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float* in_ptr =
                        &in[(oy + ky) * in_stride + (ox + kx)];

                    __m128 input = _mm_loadu_ps(in_ptr);
                    __m128 kernel =
                        _mm_set1_ps(ker[ky * K + kx]);

                    acc = _mm_add_ps(
                        _mm_mul_ps(input, kernel), acc);
                }
            }

            _mm_storeu_ps(out + oy * W + ox, acc);
        }
    }
}
EOF

build

for N in "${SIZES[@]}"; do
    timing="$OUT/raw/simd128_${N}.txt"
    perfraw="$OUT/raw/perf_simd128_${N}.txt"

    ./bin/conv simd "$N" "$N" 3 > "$timing"
    naive_ms=$(get_time "$timing" naive)
    simd_ms=$(get_time "$timing" simd)
    speed=$(awk -v a="$naive_ms" -v b="$simd_ms" 'BEGIN{printf "%.6f",a/b}')

    perf stat -r 3 -e instructions,L1-dcache-loads,L1-dcache-load-misses \
        ./bin/conv simd "$N" "$N" 3 >/dev/null 2>"$perfraw" || true

    ins=$(perf_value "$perfraw" "instructions")
    l1loads=$(perf_value "$perfraw" "L1-dcache-loads")
    l1miss=$(perf_value "$perfraw" "L1-dcache-load-misses")
    mpki=$(awk -v m="${l1miss:-0}" -v i="${ins:-0}" \
        'BEGIN{if(i>0) printf "%.6f",1000*m/i; else print "NA"}')

    echo "$N,128,$simd_ms,$speed,${ins:-NA},${l1loads:-NA},${l1miss:-NA},$mpki" >> "$OUT/task1C.csv"
done

# Restore user's original sources/build.
restore

echo
echo "============================================================"
echo "DONE"
echo "Results:"
echo "  $OUT/task1_stages.csv"
echo "  $OUT/task1B.csv"
echo "  $OUT/task1C.csv"
echo
echo "Your CPU is AVX2/FMA but not AVX-512, so a genuine 512-bit SIMD"
echo "measurement is not possible on this machine."
echo
echo "IMPORTANT: perf values collected around ./bin/conv are"
echo "process-level because the supplied harness runs the reference"
echo "and the selected stage, plus correctness/timing machinery."
echo "============================================================"
echo "============================================================"