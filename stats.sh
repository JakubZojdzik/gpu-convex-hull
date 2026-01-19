#!/usr/bin/env bash

set -e

BLOCK_SIZES=(64 128 256 384 512)
SEEDS=(123456ULL 789012ULL 345987ULL)

BLOCK_FILES=(
    gpu_quickhull_naive.cu
    gpu_quickhull.cu
)

MAIN_FILE=main.cu

for BLOCK in "${BLOCK_SIZES[@]}"; do
    echo "=============================="
    echo "Using BLOCK_SIZE = $BLOCK"
    echo "=============================="

    # Update BLOCK_SIZE in the CUDA files
    for FILE in "${BLOCK_FILES[@]}"; do
        sed -i "s/^#define BLOCK_SIZE .*/#define BLOCK_SIZE ${BLOCK}/" "$FILE"
    done

    for SEED in "${SEEDS[@]}"; do
        echo "------------------------------"
        echo "Using SEED = $SEED"
        echo "------------------------------"

        # Update seed in main.cu
        sed -i \
            "s/generate_points<<<blocks, threads>>>(d_px, d_py, N, .*ULL);/generate_points<<<blocks, threads>>>(d_px, d_py, N, ${SEED});/" \
            "$MAIN_FILE"

        make $> /dev/null

        echo "[./main]"
        ./main
        echo
    done
done
