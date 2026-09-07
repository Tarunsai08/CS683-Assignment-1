#!/usr/bin/env python3
from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt

OUT = Path("task1_results")
PLOT = OUT / "plots"
PLOT.mkdir(parents=True, exist_ok=True)

def read_csv(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))

# -------------------------------
# Task 1A / 1D technique comparison
# -------------------------------
rows = read_csv("task1_stages.csv")

for K in sorted(set(int(r["K"]) for r in rows)):
    rr = [r for r in rows if int(r["K"]) == K]
    rr.sort(key=lambda r: int(r["size"]))
    sizes = [int(r["size"]) for r in rr]

    plt.figure(figsize=(8, 5))
    for col, label in [
        ("reorder_speedup", "Loop reordering"),
        ("unroll_speedup", "Loop unrolling"),
        ("tile_speedup", "Tiling"),
        ("simd_speedup", "SIMD"),
        ("optimized_speedup", "Tiling + SIMD / optimized"),
    ]:
        plt.plot(sizes, [float(r[col]) for r in rr],
                 marker="o", label=label)
    plt.xlabel("Matrix size N (N×N)")
    plt.ylabel("Speedup vs naive")
    plt.title(f"Task 1D — Optimization comparison, K={K}")
    plt.xticks(sizes)
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT / f"task1D_speedup_K{K}.png", dpi=220)
    plt.close()

# -------------------------------
# Task 1B — speedup vs matrix size
# Choose the tile sizes present in the data.
# -------------------------------
rows = read_csv("task1B.csv")
tiles = sorted(set(int(r["tile"]) for r in rows))
sizes = sorted(set(int(r["size"]) for r in rows))

plt.figure(figsize=(9, 5))
for T in tiles:
    rr = [r for r in rows if int(r["tile"]) == T]
    rr.sort(key=lambda r: int(r["size"]))
    xs = [int(r["size"]) for r in rr]
    ys = [float(r["tile_speedup"]) for r in rr]
    plt.plot(xs, ys, marker="o", label=f"TILE={T}")
plt.xlabel("Matrix size N (N×N)")
plt.ylabel("Speedup vs naive")
plt.title("Task 1B — Tiled convolution speedup vs matrix size")
plt.xticks(sizes)
plt.grid(True, alpha=.3)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(PLOT / "task1B_speedup_vs_size.png", dpi=220)
plt.close()

# -------------------------------
# Task 1B — L1-D MPKI vs matrix size
# -------------------------------
plt.figure(figsize=(9, 5))
for T in tiles:
    rr = [r for r in rows if int(r["tile"]) == T]
    rr.sort(key=lambda r: int(r["size"]))
    xs = []
    ys = []
    for r in rr:
        if r.get("tile_MPKI", "NA") != "NA":
            xs.append(int(r["size"]))
            ys.append(float(r["tile_MPKI"]))
    if ys:
        plt.plot(xs, ys, marker="o", label=f"TILE={T}")
plt.xlabel("Matrix size N (N×N)")
plt.ylabel("L1-D MPKI")
plt.title("Task 1B — L1-D MPKI vs matrix size")
plt.xticks(sizes)
plt.grid(True, alpha=.3)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(PLOT / "task1B_L1D_MPKI_vs_size.png", dpi=220)
plt.close()

# -------------------------------
# Task 1C — SIMD width comparison
# -------------------------------
rows = read_csv("task1C.csv")
widths = sorted(set(int(r["width_bits"]) for r in rows))

plt.figure(figsize=(8, 5))
for w in widths:
    rr = [r for r in rows if int(r["width_bits"]) == w]
    rr.sort(key=lambda r: int(r["size"]))
    plt.plot([int(r["size"]) for r in rr],
             [float(r["speedup"]) for r in rr],
             marker="o", label=f"{w}-bit")
plt.xlabel("Matrix size N (N×N)")
plt.ylabel("Speedup vs naive")
plt.title("Task 1C — SIMD speedup vs matrix size")
plt.xticks(sorted(set(int(r["size"]) for r in rows)))
plt.grid(True, alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT / "task1C_speedup_vs_size_width.png", dpi=220)
plt.close()

# Instruction count plot, if available.
plt.figure(figsize=(8, 5))
for w in widths:
    rr = [r for r in rows if int(r["width_bits"]) == w and r["instructions"] != "NA"]
    rr.sort(key=lambda r: int(r["size"]))
    if rr:
        plt.plot([int(r["size"]) for r in rr],
                 [float(r["instructions"]) for r in rr],
                 marker="o", label=f"{w}-bit")
plt.xlabel("Matrix size N (N×N)")
plt.ylabel("Retired instructions (process-level perf)")
plt.title("Task 1C — Instruction count vs matrix size")
plt.xticks(sorted(set(int(r["size"]) for r in rows)))
plt.grid(True, alpha=.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT / "task1C_instructions_vs_size.png", dpi=220)
plt.close()

print("Plots written to", PLOT)