#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include "utils.h"

#define BLOCK_SIZE 512

// #define DEBUG_PRINT
#ifdef DEBUG_PRINT
#define debug(...) printf(__VA_ARGS__);
#else
#define debug(...) do {} while (0)
#endif


struct MinMaxPoint {
    float x;
    float y;
    int   idx;
};

struct MinXOp {
    __host__ __device__
    MinMaxPoint operator()(const MinMaxPoint &a,
                           const MinMaxPoint &b) const
    {
        if (a.x < b.x) return a;
        if (a.x > b.x) return b;
        return (a.y < b.y) ? a : b;
    }
};

struct MaxXOp {
    __host__ __device__
    MinMaxPoint operator()(const MinMaxPoint &a,
                           const MinMaxPoint &b) const
    {
        if (a.x > b.x) return a;
        if (a.x < b.x) return b;
        return (a.y > b.y) ? a : b;
    }
};

static __global__ void buildPointArray(const float *px,
                                const float *py,
                                MinMaxPoint *out,
                                int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x   = px[i];
        out[i].y   = py[i];
        out[i].idx = i;
    }
}

static void findMinMaxX_CUB(const float *d_px,
                     const float *d_py,
                     int n,
                     MinMaxPoint &h_min,
                     MinMaxPoint &h_max)
{
    MinMaxPoint *d_points;
    cudaMalloc(&d_points, n * sizeof(MinMaxPoint));

    int grid  = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildPointArray<<<grid, BLOCK_SIZE>>>(d_px, d_py, d_points, n);

    MinMaxPoint *d_min, *d_max;
    cudaMalloc(&d_min, sizeof(MinMaxPoint));
    cudaMalloc(&d_max, sizeof(MinMaxPoint));

    void *d_temp = nullptr;
    size_t temp_bytes = 0;

    cub::DeviceReduce::Reduce(
        d_temp, temp_bytes,
        d_points, d_min,
        n,
        MinXOp(),
        MinMaxPoint{FLT_MAX, FLT_MAX, -1}
    );

    cudaMalloc(&d_temp, temp_bytes);

    cub::DeviceReduce::Reduce(
        d_temp, temp_bytes,
        d_points, d_min,
        n,
        MinXOp(),
        MinMaxPoint{FLT_MAX, FLT_MAX, -1}
    );

    cub::DeviceReduce::Reduce(
        d_temp, temp_bytes,
        d_points, d_max,
        n,
        MaxXOp(),
        MinMaxPoint{-FLT_MAX, -FLT_MAX, -1}
    );

    cudaMemcpy(&h_min, d_min, sizeof(MinMaxPoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(MinMaxPoint), cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_points);
    cudaFree(d_min);
    cudaFree(d_max);
}

// Pair of distance and point index
struct DistIdxPairPaper {
    float dist;
    int   idx;
};

struct MaxDistOpPaper {
    __host__ __device__
    DistIdxPairPaper operator()(const DistIdxPairPaper &a, const DistIdxPairPaper &b) const {
        return (a.dist >= b.dist) ? a : b;
    }
};

static __global__ void buildDistIdxArrayPaper(const float *distances, DistIdxPairPaper *pairs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pairs[i].dist = distances[i];
        pairs[i].idx = i;
    }
}

// GPU kernel to find segment offsets - no host involvement
static __global__ void findSegmentOffsetsKernelPaper(const int *labels, int *offsets, int numSegments, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i == 0) {
        offsets[0] = 0;  // First segment always starts at 0
        offsets[numSegments] = n;  // Sentinel
    }
    
    if (i < n - 1 && labels[i] != labels[i + 1]) {
        offsets[labels[i + 1]] = i + 1;
    }
}

// GPU kernel to fix empty segments (propagate offsets backwards)
static __global__ void fixEmptySegmentsKernel(int *offsets, int numSegments) {
    // Single thread fixes empty segments - for small numSegments this is fine
    // For large numSegments, a parallel scan-based approach would be better
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = numSegments - 1; i >= 1; i--) {
            if (offsets[i] == -1) {
                offsets[i] = offsets[i + 1];
            }
        }
    }
}

static void segmentedMaxDistReducePaper(
    const float *d_distances,
    const int *d_labels,
    int *d_segmentOffsets,
    DistIdxPairPaper *d_maxPerSegment,
    DistIdxPairPaper *d_pairs,
    void *d_tempStorage,
    size_t tempStorageBytes,
    int n,
    int numSegments)
{
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Initialize offsets to -1 for empty segment detection
    cudaMemset(d_segmentOffsets, -1, (numSegments + 1) * sizeof(int));
    
    findSegmentOffsetsKernelPaper<<<numBlocks, BLOCK_SIZE>>>(
        d_labels, d_segmentOffsets, numSegments, n);
    
    // Fix empty segments entirely on GPU
    fixEmptySegmentsKernel<<<1, 1>>>(d_segmentOffsets, numSegments);
    
    buildDistIdxArrayPaper<<<numBlocks, BLOCK_SIZE>>>(d_distances, d_pairs, n);
    
    DistIdxPairPaper identity{0.0f, -1};
    
    cub::DeviceSegmentedReduce::Reduce(
        d_tempStorage, tempStorageBytes,
        d_pairs, d_maxPerSegment,
        numSegments,
        d_segmentOffsets, d_segmentOffsets + 1,
        MaxDistOpPaper(), identity);
}


static __global__ void computeDistancesKernelPaper(float *px, float *py, int *labels,
                                        float *ansX, float *ansY, int ansSize,
                                        float *distances, int n) {
    // Shared memory for ANS array - only load what's needed for this block
    extern __shared__ float sharedAns[];
    float *sAnsX = sharedAns;
    float *sAnsY = &sharedAns[BLOCK_SIZE + 2];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ int minLabel, maxLabel;

    if (tid == 0) {
        minLabel = ansSize;
        maxLabel = -1;
    }
    __syncthreads();

    if (idx < n) {
        int label = labels[idx];
        atomicMin(&minLabel, label);
        atomicMax(&maxLabel, label);
    }
    __syncthreads();

    // Load required ANS entries into shared memory
    // Only load [minLabel, maxLabel+1] range from ANS
    int ansRange = maxLabel - minLabel + 2;
    for (int i = tid; i < ansRange; i += blockDim.x) {
        int ansIdx = minLabel + i;
        if (ansIdx < ansSize) {
            sAnsX[i] = ansX[ansIdx];
            sAnsY[i] = ansY[ansIdx];
        }
    }
    __syncthreads();

    if (idx >= n) return;

    int label = labels[idx];
    int localLabel = label - minLabel;

    float lx = sAnsX[localLabel];
    float ly = sAnsY[localLabel];
    float rx = sAnsX[localLabel + 1];
    float ry = sAnsY[localLabel + 1];

    float curX = px[idx];
    float curY = py[idx];

    // Cross product gives signed distance
    float d = (rx - lx) * (curY - ly) - (ry - ly) * (curX - lx);
    distances[idx] = d;
}

// Determine which side each point goes to (left or right of max point)
// Also marks the max point for each segment
static __global__ void determineSideKernel(float *px, float *py, int *labels,
                                    float *ansX, float *ansY, 
                                    DistIdxPairPaper *maxIdxPerLabel,
                                    int *goesLeft, int *goesRight, 
                                    int *isMaxPoint, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = labels[idx];
    int maxIdx = maxIdxPerLabel[label].idx;
    
    // No valid max point for this segment - discard all points
    if (maxIdx < 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        isMaxPoint[idx] = 0;
        return;
    }

    // This is the max point - mark it but don't include in either partition
    if (maxIdx == idx) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        isMaxPoint[idx] = 1;
        return;
    }
    
    isMaxPoint[idx] = 0;

    float lx = ansX[label];
    float ly = ansY[label];
    float rx = ansX[label + 1];
    float ry = ansY[label + 1];
    float mx = px[maxIdx];
    float my = py[maxIdx];
    float curX = px[idx];
    float curY = py[idx];
    
    // Distance from line L-M (left partition)
    float distLM = (mx - lx) * (curY - ly) - (my - ly) * (curX - lx);
    // Distance from line M-R (right partition)
    float distMR = (rx - mx) * (curY - my) - (ry - my) * (curX - mx);
    
    if (distLM > 0) {
        goesLeft[idx] = 1;
        goesRight[idx] = 0;
    } else if (distMR > 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 1;
    } else {
        // Point is inside the triangle or on the edge - discard
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
    }
}

// Compute state array: state[label] = 1 if label has a valid max point
static __global__ void computeStateKernel(DistIdxPairPaper *maxIdxPerLabel, int *state, int numLabels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLabels) return;
    
    state[i] = (maxIdxPerLabel[i].idx >= 0 && maxIdxPerLabel[i].dist > 0) ? 1 : 0;
}

// Key insight from paper: use head flags for segmented scan
// Head flag is 1 at the start of each segment
static __global__ void computeHeadFlagsKernel(int *labels, int *headFlags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (idx == 0) {
        headFlags[idx] = 1;
    } else {
        headFlags[idx] = (labels[idx] != labels[idx - 1]) ? 1 : 0;
    }
}

// Segmented prefix sum using head flags
// This computes prefix sum that resets at each segment boundary
struct SegmentedSumOp {
    __host__ __device__
    int2 operator()(const int2 &a, const int2 &b) const {
        // a.x = value, a.y = head flag
        // If b has head flag, start fresh from b
        if (b.y) {
            return b;
        }
        return make_int2(a.x + b.x, a.y);
    }
};

// Compute the new label for each point based on prefix sums
// Paper approach 7b-7c: use prefix scans to compute output positions
static __global__ void computeNewLabelsAndPositionsKernel(
    int *labels, int *statePrefixSum,
    int *goesLeft, int *goesRight,
    int *leftScanResult, int *rightScanResult,
    int *leftSegmentCounts, int *rightSegmentCounts,
    int *newLabels, int *outputPositions, int *keepFlags,
    int numLabels, int n) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = labels[idx];
    int keep = goesLeft[idx] | goesRight[idx];
    keepFlags[idx] = keep;
    
    if (!keep) {
        newLabels[idx] = -1;
        outputPositions[idx] = -1;
        return;
    }
    
    // New label calculation from paper:
    // Left partition gets label: oldLabel + statePrefixSum[oldLabel]
    // Right partition gets label: oldLabel + statePrefixSum[oldLabel] + 1
    int baseNewLabel = label + statePrefixSum[label];
    
    if (goesLeft[idx]) {
        newLabels[idx] = baseNewLabel;
        // Position within segment from segmented scan
        outputPositions[idx] = leftScanResult[idx];
    } else {
        newLabels[idx] = baseNewLabel + 1;
        // Position within segment from segmented scan
        // Offset by left count for this label
        outputPositions[idx] = rightScanResult[idx];
    }
}

// Compute per-segment counts using the last value in each segment
static __global__ void extractSegmentCountsKernel(
    int *goesLeft, int *goesRight,
    int *leftScan, int *rightScan,
    int *labels,
    int *leftCounts, int *rightCounts,
    int numLabels, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = labels[idx];
    
    // Check if this is the last element of its segment
    bool isLast = (idx == n - 1) || (labels[idx + 1] != label);
    
    if (isLast) {
        leftCounts[label] = leftScan[idx] + goesLeft[idx];
        rightCounts[label] = rightScan[idx] + goesRight[idx];
    }
}

// Compute output offsets for each new label
static __global__ void computeLabelOffsetsKernel(
    int *leftCounts, int *rightCounts,
    int *state, int *statePrefixSum,
    int *labelOffsets,
    int numLabels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLabels) return;
    
    int baseNewLabel = i + statePrefixSum[i];
    
    // Left partition offset
    labelOffsets[baseNewLabel] = 0;  // Will be computed via prefix sum
    
    // Right partition offset (if state[i] == 1, meaning partition splits)
    if (state[i]) {
        labelOffsets[baseNewLabel + 1] = 0;
    }
}

// Final scatter kernel - moves points to their new positions
static __global__ void scatterPointsKernel(
    float *pxIn, float *pyIn, int *labelsIn,
    float *pxOut, float *pyOut, int *labelsOut,
    int *newLabels, int *outputPositions, int *keepFlags,
    int *labelWriteOffsets,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (!keepFlags[idx]) return;
    
    int newLabel = newLabels[idx];
    int posInSegment = outputPositions[idx];
    int outIdx = labelWriteOffsets[newLabel] + posInSegment;
    
    pxOut[outIdx] = pxIn[idx];
    pyOut[outIdx] = pyIn[idx];
    labelsOut[outIdx] = newLabel;
}

// Build new ANS array with inserted max points
static __global__ void buildNewAnsKernelPaper(float *oldAnsX, float *oldAnsY,
                                   float *newAnsX, float *newAnsY,
                                   float *px, float *py,
                                   int *state, int *statePrefixSum,
                                   DistIdxPairPaper *maxPerSegment,
                                   int numLabels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > numLabels) return;
    
    if (i < numLabels) {
        int newPos = i + statePrefixSum[i];
        newAnsX[newPos] = oldAnsX[i];
        newAnsY[newPos] = oldAnsY[i];
        
        // If partition splits, insert max point
        if (state[i] == 1 && maxPerSegment[i].idx >= 0) {
            int maxIdx = maxPerSegment[i].idx;
            newAnsX[newPos + 1] = px[maxIdx];
            newAnsY[newPos + 1] = py[maxIdx];
        }
    } else {
        // i == numLabels: copy final endpoint
        int totalInserted = statePrefixSum[numLabels - 1] + state[numLabels - 1];
        int newPos = numLabels + totalInserted;
        newAnsX[newPos] = oldAnsX[numLabels];
        newAnsY[newPos] = oldAnsY[numLabels];
    }
}

// Helper for CUB exclusive scan
static void cubExclusiveScanIntPaper(int *d_input, int *d_output, int n, 
                                 void *d_temp, size_t tempBytes) {
    cub::DeviceScan::ExclusiveSum(d_temp, tempBytes, d_input, d_output, n);
}

// Segmented exclusive scan using CUB
// Uses head flags to reset sum at segment boundaries
static __global__ void prepareForSegmentedScan(int *values, int *headFlags, int2 *pairs, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    pairs[idx] = make_int2(values[idx], headFlags[idx]);
}

static __global__ void extractSegmentedScanResult(int2 *pairs, int *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    result[idx] = pairs[idx].x;
}

// Custom segmented scan implementation using head flags
// For each segment, computes exclusive prefix sum that resets at boundaries
static void segmentedExclusiveScan(int *d_values, int *d_headFlags, int *d_output,
                            int2 *d_pairs, void *d_temp, size_t tempBytes, int n) {
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Prepare pairs (value, headFlag)
    prepareForSegmentedScan<<<numBlocks, BLOCK_SIZE>>>(d_values, d_headFlags, d_pairs, n);
    
    // Run segmented scan with custom operator
    cub::DeviceScan::ExclusiveScan(d_temp, tempBytes, d_pairs, d_pairs, 
                                    SegmentedSumOp(), make_int2(0, 0), n);
    
    // Extract results
    extractSegmentedScanResult<<<numBlocks, BLOCK_SIZE>>>(d_pairs, d_output, n);
}

// Pre-allocated workspace structure
struct QuickHullWorkspace {
    // Point data (double buffered)
    float *d_px[2], *d_py[2];
    int *d_labels[2];
    
    // Per-point arrays
    float *d_distances;
    int *d_goesLeft, *d_goesRight;
    int *d_headFlags;
    int *d_leftScan, *d_rightScan;
    int *d_newLabels, *d_outputPositions, *d_keepFlags;
    int *d_isMaxPoint;
    int2 *d_scanPairs;
    
    // Per-segment arrays
    int *d_segmentOffsets;
    DistIdxPairPaper *d_maxPerSegment;
    DistIdxPairPaper *d_pairs;
    int *d_state, *d_statePrefixSum;
    int *d_leftCounts, *d_rightCounts;
    int *d_labelOffsets, *d_labelWriteOffsets;
    
    // ANS arrays
    float *d_ansX, *d_ansY;
    float *d_newAnsX, *d_newAnsY;
    
    // Temp storage for CUB
    void *d_tempStorage;
    size_t tempStorageBytes;
    
    int maxN;
    int maxLabels;
};

void allocateWorkspace(QuickHullWorkspace &ws, int n) {
    ws.maxN = n;
    ws.maxLabels = n + 1;  // At most n labels
    
    // Double-buffered point arrays
    cudaMalloc(&ws.d_px[0], n * sizeof(float));
    cudaMalloc(&ws.d_py[0], n * sizeof(float));
    cudaMalloc(&ws.d_labels[0], n * sizeof(int));
    cudaMalloc(&ws.d_px[1], n * sizeof(float));
    cudaMalloc(&ws.d_py[1], n * sizeof(float));
    cudaMalloc(&ws.d_labels[1], n * sizeof(int));
    
    // Per-point arrays
    cudaMalloc(&ws.d_distances, n * sizeof(float));
    cudaMalloc(&ws.d_goesLeft, n * sizeof(int));
    cudaMalloc(&ws.d_goesRight, n * sizeof(int));
    cudaMalloc(&ws.d_headFlags, n * sizeof(int));
    cudaMalloc(&ws.d_leftScan, n * sizeof(int));
    cudaMalloc(&ws.d_rightScan, n * sizeof(int));
    cudaMalloc(&ws.d_newLabels, n * sizeof(int));
    cudaMalloc(&ws.d_outputPositions, n * sizeof(int));
    cudaMalloc(&ws.d_keepFlags, n * sizeof(int));
    cudaMalloc(&ws.d_isMaxPoint, n * sizeof(int));
    cudaMalloc(&ws.d_scanPairs, n * sizeof(int2));
    cudaMalloc(&ws.d_pairs, n * sizeof(DistIdxPairPaper));
    
    // Per-segment arrays
    cudaMalloc(&ws.d_segmentOffsets, (ws.maxLabels + 1) * sizeof(int));
    cudaMalloc(&ws.d_maxPerSegment, ws.maxLabels * sizeof(DistIdxPairPaper));
    cudaMalloc(&ws.d_state, ws.maxLabels * sizeof(int));
    cudaMalloc(&ws.d_statePrefixSum, ws.maxLabels * sizeof(int));
    cudaMalloc(&ws.d_leftCounts, ws.maxLabels * sizeof(int));
    cudaMalloc(&ws.d_rightCounts, ws.maxLabels * sizeof(int));
    cudaMalloc(&ws.d_labelOffsets, (ws.maxLabels + 1) * sizeof(int));
    cudaMalloc(&ws.d_labelWriteOffsets, (ws.maxLabels + 1) * sizeof(int));
    
    // ANS arrays
    cudaMalloc(&ws.d_ansX, (n + 2) * sizeof(float));
    cudaMalloc(&ws.d_ansY, (n + 2) * sizeof(float));
    cudaMalloc(&ws.d_newAnsX, (n + 2) * sizeof(float));
    cudaMalloc(&ws.d_newAnsY, (n + 2) * sizeof(float));
    
    // Determine temp storage size for CUB operations
    ws.d_tempStorage = nullptr;
    ws.tempStorageBytes = 0;
    
    // Get max temp storage needed across all CUB operations
    size_t tempBytes1 = 0, tempBytes2 = 0, tempBytes3 = 0;
    
    cub::DeviceScan::ExclusiveSum(nullptr, tempBytes1, ws.d_state, ws.d_statePrefixSum, ws.maxLabels);
    cub::DeviceScan::ExclusiveScan(nullptr, tempBytes2, ws.d_scanPairs, ws.d_scanPairs, 
                                    SegmentedSumOp(), make_int2(0, 0), n);
    
    DistIdxPairPaper identity{0.0f, -1};
    cub::DeviceSegmentedReduce::Reduce(nullptr, tempBytes3, ws.d_pairs, ws.d_maxPerSegment,
                                        ws.maxLabels, ws.d_segmentOffsets, ws.d_segmentOffsets + 1,
                                        MaxDistOpPaper(), identity);
    
    ws.tempStorageBytes = std::max({tempBytes1, tempBytes2, tempBytes3});
    cudaMalloc(&ws.d_tempStorage, ws.tempStorageBytes);
}

void freeWorkspace(QuickHullWorkspace &ws) {
    cudaFree(ws.d_px[0]); cudaFree(ws.d_py[0]); cudaFree(ws.d_labels[0]);
    cudaFree(ws.d_px[1]); cudaFree(ws.d_py[1]); cudaFree(ws.d_labels[1]);
    cudaFree(ws.d_distances);
    cudaFree(ws.d_goesLeft); cudaFree(ws.d_goesRight);
    cudaFree(ws.d_headFlags);
    cudaFree(ws.d_leftScan); cudaFree(ws.d_rightScan);
    cudaFree(ws.d_newLabels); cudaFree(ws.d_outputPositions); cudaFree(ws.d_keepFlags);
    cudaFree(ws.d_isMaxPoint);
    cudaFree(ws.d_scanPairs);
    cudaFree(ws.d_pairs);
    cudaFree(ws.d_segmentOffsets);
    cudaFree(ws.d_maxPerSegment);
    cudaFree(ws.d_state); cudaFree(ws.d_statePrefixSum);
    cudaFree(ws.d_leftCounts); cudaFree(ws.d_rightCounts);
    cudaFree(ws.d_labelOffsets); cudaFree(ws.d_labelWriteOffsets);
    cudaFree(ws.d_ansX); cudaFree(ws.d_ansY);
    cudaFree(ws.d_newAnsX); cudaFree(ws.d_newAnsY);
    cudaFree(ws.d_tempStorage);
}

// Compute segment write offsets from per-segment counts
static __global__ void computeSegmentWriteOffsetsKernel(
    int *leftCounts, int *rightCounts,
    int *state, int *statePrefixSum,
    int *writeOffsets,
    int numLabels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLabels) return;
    
    int newLabelBase = i + statePrefixSum[i];
    
    // This will be prefix-summed later
    writeOffsets[newLabelBase] = leftCounts[i];
    if (state[i]) {
        writeOffsets[newLabelBase + 1] = rightCounts[i];
    }
}

static void gpuQuickHullOneSide(float *h_px, float *h_py, int n,
                          float leftX, float leftY, float rightX, float rightY,
                          std::vector<Point> &hullPoints) {
    if (n == 0) return;

    // Allocate workspace once
    QuickHullWorkspace ws;
    allocateWorkspace(ws, n);
    
    // Initialize
    int currentBuffer = 0;
    float *d_px = ws.d_px[currentBuffer];
    float *d_py = ws.d_py[currentBuffer];
    int *d_labels = ws.d_labels[currentBuffer];
    
    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_labels, 0, n * sizeof(int));
    
    // Initialize ANS with endpoints
    float h_ansInit[2] = {leftX, rightX};
    float h_ansInitY[2] = {leftY, rightY};
    cudaMemcpy(ws.d_ansX, h_ansInit, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ws.d_ansY, h_ansInitY, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    int currentN = n;
    int numLabels = 1;
    int ansSize = 2;
    
    while (currentN > 0) {
        int numBlocks = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int labelBlocks = (numLabels + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Step 1: Compute distances for all points
        size_t sharedMemSize = 2 * (BLOCK_SIZE + 2) * sizeof(float);
        computeDistancesKernelPaper<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_labels,
            ws.d_ansX, ws.d_ansY, ansSize,
            ws.d_distances, currentN);
        
        // Step 2: Segmented max reduction to find max point per segment
        segmentedMaxDistReducePaper(ws.d_distances, d_labels, ws.d_segmentOffsets,
                               ws.d_maxPerSegment, ws.d_pairs, 
                               ws.d_tempStorage, ws.tempStorageBytes,
                               currentN, numLabels);
        
        // Step 3: Compute state array (1 if segment has valid max point)
        computeStateKernel<<<labelBlocks, BLOCK_SIZE>>>(
            ws.d_maxPerSegment, ws.d_state, numLabels);
        
        // Step 4: Prefix sum on state to get insertion positions
        cub::DeviceScan::ExclusiveSum(ws.d_tempStorage, ws.tempStorageBytes,
                                       ws.d_state, ws.d_statePrefixSum, numLabels);
        
        // Check if any new hull points found
        int lastState, lastStatePrefixSum;
        cudaMemcpy(&lastState, ws.d_state + numLabels - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastStatePrefixSum, ws.d_statePrefixSum + numLabels - 1, sizeof(int), cudaMemcpyDeviceToHost);
        int numNewHullPoints = lastState + lastStatePrefixSum;
        
        if (numNewHullPoints == 0) {
            break;  // No more hull points to find
        }
        
        int newNumLabels = numLabels + numNewHullPoints;
        
        // Step 5: Determine which side each point goes to
        cudaMemset(ws.d_goesLeft, 0, currentN * sizeof(int));
        cudaMemset(ws.d_goesRight, 0, currentN * sizeof(int));
        determineSideKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels,
            ws.d_ansX, ws.d_ansY,
            ws.d_maxPerSegment,
            ws.d_goesLeft, ws.d_goesRight, ws.d_isMaxPoint, currentN);
        
        // Step 6: Compute head flags for segmented scan
        computeHeadFlagsKernel<<<numBlocks, BLOCK_SIZE>>>(d_labels, ws.d_headFlags, currentN);
        
        // Step 7a-7c: Use segmented prefix scans to compute output positions
        // 7a: Segmented scan of goesLeft
        segmentedExclusiveScan(ws.d_goesLeft, ws.d_headFlags, ws.d_leftScan,
                               ws.d_scanPairs, ws.d_tempStorage, ws.tempStorageBytes, currentN);
        
        // 7b: Segmented scan of goesRight
        segmentedExclusiveScan(ws.d_goesRight, ws.d_headFlags, ws.d_rightScan,
                               ws.d_scanPairs, ws.d_tempStorage, ws.tempStorageBytes, currentN);
        
        // Extract per-segment counts
        cudaMemset(ws.d_leftCounts, 0, numLabels * sizeof(int));
        cudaMemset(ws.d_rightCounts, 0, numLabels * sizeof(int));
        extractSegmentCountsKernel<<<numBlocks, BLOCK_SIZE>>>(
            ws.d_goesLeft, ws.d_goesRight,
            ws.d_leftScan, ws.d_rightScan,
            d_labels,
            ws.d_leftCounts, ws.d_rightCounts,
            numLabels, currentN);
        
        // Compute new labels for each point
        computeNewLabelsAndPositionsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, ws.d_statePrefixSum,
            ws.d_goesLeft, ws.d_goesRight,
            ws.d_leftScan, ws.d_rightScan,
            ws.d_leftCounts, ws.d_rightCounts,
            ws.d_newLabels, ws.d_outputPositions, ws.d_keepFlags,
            numLabels, currentN);
        
        // Compute write offsets for each new label
        cudaMemset(ws.d_labelOffsets, 0, (newNumLabels + 1) * sizeof(int));
        computeSegmentWriteOffsetsKernel<<<labelBlocks, BLOCK_SIZE>>>(
            ws.d_leftCounts, ws.d_rightCounts,
            ws.d_state, ws.d_statePrefixSum,
            ws.d_labelOffsets, numLabels);
        
        // Prefix sum to get actual write offsets
        cub::DeviceScan::ExclusiveSum(ws.d_tempStorage, ws.tempStorageBytes,
                                       ws.d_labelOffsets, ws.d_labelWriteOffsets, newNumLabels + 1);
        
        // Get new point count
        int newN;
        cudaMemcpy(&newN, ws.d_labelWriteOffsets + newNumLabels, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Build new ANS array
        int newAnsSize = ansSize + numNewHullPoints;
        int ansBlocks = (numLabels + 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        buildNewAnsKernelPaper<<<ansBlocks, BLOCK_SIZE>>>(
            ws.d_ansX, ws.d_ansY, ws.d_newAnsX, ws.d_newAnsY,
            d_px, d_py, ws.d_state, ws.d_statePrefixSum, ws.d_maxPerSegment, numLabels);
        
        // Swap ANS buffers
        std::swap(ws.d_ansX, ws.d_newAnsX);
        std::swap(ws.d_ansY, ws.d_newAnsY);
        ansSize = newAnsSize;
        
        if (newN == 0) {
            break;
        }
        
        // Scatter points to new positions
        int nextBuffer = 1 - currentBuffer;
        scatterPointsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels,
            ws.d_px[nextBuffer], ws.d_py[nextBuffer], ws.d_labels[nextBuffer],
            ws.d_newLabels, ws.d_outputPositions, ws.d_keepFlags,
            ws.d_labelWriteOffsets, currentN);
        
        // Switch to new buffer
        currentBuffer = nextBuffer;
        d_px = ws.d_px[currentBuffer];
        d_py = ws.d_py[currentBuffer];
        d_labels = ws.d_labels[currentBuffer];
        
        currentN = newN;
        numLabels = newNumLabels;
    }
    
    // Copy result back
    float *h_ansX = (float*)malloc(ansSize * sizeof(float));
    float *h_ansY = (float*)malloc(ansSize * sizeof(float));
    cudaMemcpy(h_ansX, ws.d_ansX, ansSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ansY, ws.d_ansY, ansSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Add hull points (excluding endpoints which are added by caller)
    for (int i = 1; i < ansSize - 1; i++) {
        hullPoints.push_back({h_ansX[i], h_ansY[i]});
    }
    
    free(h_ansX);
    free(h_ansY);
    freeWorkspace(ws);
}


extern "C" void gpuQuickHullPaper(float *h_px, float *h_py, int n,
                              float *result_x, float *result_y, int *M) {
    if (n <= 2) {
        for (int i = 0; i < n; i++) {
            result_x[i] = h_px[i];
            result_y[i] = h_py[i];
        }
        *M = n;
        return;
    }

    float *d_px, *d_py;
    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);

    MinMaxPoint h_min, h_max;
    findMinMaxX_CUB(d_px, d_py, n, h_min, h_max);

    cudaFree(d_px);
    cudaFree(d_py);

    Point minPt = {h_min.x, h_min.y};
    Point maxPt = {h_max.x, h_max.y};

    std::vector<float> upperX, upperY, lowerX, lowerY;
    upperX.reserve(n);
    upperY.reserve(n);
    lowerX.reserve(n);
    lowerY.reserve(n);

    for (int i = 0; i < n; i++) {
        float d = (maxPt.x - minPt.x) * (h_py[i] - minPt.y) - 
                  (maxPt.y - minPt.y) * (h_px[i] - minPt.x);
        if (d > 0) { // upper hull
            upperX.push_back(h_px[i]);
            upperY.push_back(h_py[i]);
        } else if (d < 0) { // lower hull
            lowerX.push_back(h_px[i]);
            lowerY.push_back(h_py[i]);
        }
    }


    std::vector<Point> upperHull;
    if (!upperX.empty()) {
        gpuQuickHullOneSide(upperX.data(), upperY.data(), upperX.size(),
                            minPt.x, minPt.y, maxPt.x, maxPt.y, upperHull);
    }

    std::vector<Point> lowerHull;
    if (!lowerX.empty()) {
        gpuQuickHullOneSide(lowerX.data(), lowerY.data(), lowerX.size(),
                            maxPt.x, maxPt.y, minPt.x, minPt.y, lowerHull);
    }

    std::vector<Point> hull;
    hull.push_back(minPt);
    for (auto &p : upperHull) {
        hull.push_back(p);
    }
    hull.push_back(maxPt);
    for (auto &p : lowerHull) {
        hull.push_back(p);
    }

    *M = hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
}
