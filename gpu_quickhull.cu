#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <climits>
#include <vector>
#include <cudpp.h>
#include "utils.h"

// Block size for CUDA kernels (as per paper: chunks of 512)
#define BLOCK_SIZE 512

// Global CUDPP handle
static CUDPPHandle cudppHandle = 0;
static bool cudppInitialized = false;

void initCUDPP() {
    if (!cudppInitialized) {
        cudppCreate(&cudppHandle);
        cudppInitialized = true;
    }
}

// ============================================================================
// Steps 5-6: Find min and max X using CUDPP reduce
// Paper: "We have implemented steps 5 and 6 by using two prefix scans 
// (taking max operator once and min operator once)"
// ============================================================================
void findMinMaxWithCUDPP(float *d_px, int n, float *minX, float *maxX) {
    initCUDPP();
    
    float *d_minX, *d_maxX;
    cudaMalloc(&d_minX, sizeof(float));
    cudaMalloc(&d_maxX, sizeof(float));
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_REDUCE;
    config.datatype = CUDPP_FLOAT;
    config.options = 0;
    
    // MIN reduce
    config.op = CUDPP_MIN;
    CUDPPHandle planMin;
    cudppPlan(cudppHandle, &planMin, config, n, 1, 0);
    cudppReduce(planMin, d_minX, d_px, n);
    cudppDestroyPlan(planMin);
    
    // MAX reduce
    config.op = CUDPP_MAX;
    CUDPPHandle planMax;
    cudppPlan(cudppHandle, &planMax, config, n, 1, 0);
    cudppReduce(planMax, d_maxX, d_px, n);
    cudppDestroyPlan(planMax);
    
    cudaMemcpy(minX, d_minX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(maxX, d_maxX, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_minX);
    cudaFree(d_maxX);
}

// ============================================================================
// CUDPP-based exclusive scan for prefix sums (used in steps 32-35, 36-47)
// Paper: "We implemented steps 32-35 and 50-52 by using one prefix scan"
// ============================================================================
void cudppExclusiveScan(int *d_input, int *d_output, int n) {
    initCUDPP();
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SCAN;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.options = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
    
    CUDPPHandle plan;
    cudppPlan(cudppHandle, &plan, config, n, 1, 0);
    cudppScan(plan, d_output, d_input, n);
    cudppDestroyPlan(plan);
}

// ============================================================================
// Steps 8-16: Compute distances with shared memory optimization
// Paper: "As the points in each chunk are ordered on the basis of the label 
// values, we have loaded only the required chunk of the ANS array into the 
// shared memory"
// 
// Since points are sorted by label, consecutive threads access same/nearby labels.
// We load the relevant ANS entries into shared memory for fast access.
// ============================================================================
__global__ void computeDistancesKernel(float *px, float *py, int *labels,
                                        float *ansX, float *ansY, int ansSize,
                                        float *distances, int n) {
    // Shared memory for ANS chunk - each partition needs 2 consecutive points
    extern __shared__ float sharedAns[];
    float *sAnsX = sharedAns;
    float *sAnsY = &sharedAns[BLOCK_SIZE + 2];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Find the range of labels in this block
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
    // We need indices [minLabel, maxLabel+1] from global ANS
    // Store them at local indices [0, ansRange-1] in shared memory
    int ansRange = maxLabel - minLabel + 2;
    if (tid < ansRange && (minLabel + tid) < ansSize) {
        sAnsX[tid] = ansX[minLabel + tid];
        sAnsY[tid] = ansY[minLabel + tid];
    }
    __syncthreads();
    
    if (idx >= n) return;
    
    int label = labels[idx];
    int localLabel = label - minLabel;
    
    // Get line endpoints from shared memory (steps 11-12)
    float lx = sAnsX[localLabel];
    float ly = sAnsY[localLabel];
    float rx = sAnsX[localLabel + 1];
    float ry = sAnsY[localLabel + 1];
    
    float curX = px[idx];
    float curY = py[idx];
    
    // Compute distance (steps 13-15)
    float d = (rx - lx) * (curY - ly) - (ry - ly) * (curX - lx);
    
    distances[idx] = d;
}

// ============================================================================
// Steps 22-30: Find max distance point per partition using segmented scan
// Paper: "We have implemented steps 22-30 using matrix segmented scan approach"
// Reference: Sengupta et al. "Scan Primitives for GPU Computing"
// 
// Uses CUDPP's cudppSegmentedScan with MAX operator to find the maximum
// distance within each partition (segment).
// ============================================================================

// Create segment head flags: 1 at start of each partition, 0 elsewhere
__global__ void createSegmentFlagsKernel(int *labels, unsigned int *flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Mark segment head: first element or when label changes
    if (idx == 0 || labels[idx] != labels[idx - 1]) {
        flags[idx] = 1;
    } else {
        flags[idx] = 0;
    }
}

// After segmented max scan, find points that have the max distance in their partition
__global__ void findMaxPointFromScanKernel(float *px, float *py, int *labels,
                                            float *distances, float *scanResult,
                                            unsigned int *flags,
                                            int *maxIdx, int *state,
                                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float d = distances[idx];
    if (d <= 0) return;
    
    int label = labels[idx];
    
    // Check if this is the last element of the segment or next element starts new segment
    // Note: check idx < n-1 first to avoid out-of-bounds read of flags[n]
    bool isLastInSegment = (idx == n - 1) || (idx < n - 1 && flags[idx + 1] == 1);
    
    if (isLastInSegment) {
        // scanResult[idx] contains the max distance for this segment
        float maxDist = scanResult[idx];
        state[label] = (maxDist > 0) ? 1 : 0;
    }
    
    // Find the actual point with max distance
    // Use atomicMin to ensure only one thread wins (the one with lowest index)
    float maxInSegment = scanResult[idx];
    if (d == maxInSegment) {
        atomicMin(&maxIdx[label], idx);
    }
}

// CUDPP segmented scan for finding max per partition
void cudppSegmentedMaxScan(float *d_input, unsigned int *d_flags, float *d_output, int n) {
    printf("[DEBUG]     [cudppSegmentedMaxScan] called with n=%d\n", n); fflush(stdout);
    initCUDPP();
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SEGMENTED_SCAN;
    config.op = CUDPP_MAX;
    config.datatype = CUDPP_FLOAT;
    config.options = CUDPP_OPTION_INCLUSIVE | CUDPP_OPTION_FORWARD;
    CUDPPHandle plan;
    CUDPPResult res = cudppPlan(cudppHandle, &plan, config, n, 1, 0);
    if (res != CUDPP_SUCCESS) {
        printf("[ERROR] cudppPlan failed in cudppSegmentedMaxScan!\n"); fflush(stdout);
    }
    printf("[DEBUG]     [cudppSegmentedMaxScan] plan created\n"); fflush(stdout);
    res = cudppSegmentedScan(plan, d_output, d_input, d_flags, n);
    if (res != CUDPP_SUCCESS) {
        printf("[ERROR] cudppSegmentedScan failed in cudppSegmentedMaxScan!\n"); fflush(stdout);
    }
    printf("[DEBUG]     [cudppSegmentedMaxScan] scan done\n"); fflush(stdout);
    cudppDestroyPlan(plan);
    printf("[DEBUG]     [cudppSegmentedMaxScan] plan destroyed\n"); fflush(stdout);
}

// ============================================================================
// Steps 36-47: Compact and relabel points using prefix scans
// ============================================================================

__global__ void classifyPointsKernel(float *px, float *py, int *labels,
                                      float *distances, int *statePrefix,
                                      float *ansX, float *ansY,
                                      int *goesLeft, int *goesRight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float d = distances[idx];
    if (d <= 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        return;
    }
    
    int oldLabel = labels[idx];
    int newLabel = oldLabel + statePrefix[oldLabel];
    
    float curX = px[idx];
    float mx = ansX[newLabel + 1];  // MAX point's X coordinate
    
    int right = (curX >= mx) ? 1 : 0;
    goesLeft[idx] = 1 - right;
    goesRight[idx] = right;
}

// Create segment flags for segmented scan (1 at start of each partition)
__global__ void createPartitionFlagsKernel(int *labels, unsigned int *flags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (idx == 0 || labels[idx] != labels[idx - 1]) {
        flags[idx] = 1;
    } else {
        flags[idx] = 0;
    }
}

// Compact points to new arrays using prefix scan results
// leftScan: exclusive prefix sum of goesLeft within each segment
// rightScan: exclusive prefix sum of goesRight within each segment
__global__ void compactWithScanKernel(float *px, float *py, int *labels,
                                       int *goesLeft, int *goesRight,
                                       int *leftScan, int *rightScan,
                                       int *leftCountPerPartition, int *statePrefix,
                                       int *state, int *partitionStart,
                                       float *pxNew, float *pyNew, int *labelsNew,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Skip points that don't go anywhere (negative distance)
    if (!goesLeft[idx] && !goesRight[idx]) return;
    
    int oldLabel = labels[idx];
    int newLabelBase = oldLabel + statePrefix[oldLabel];
    
    int newLabel, posInPartition;
    
    // Check if this partition actually splits (has a MAX point)
    bool partitionSplits = (state[oldLabel] == 1);
    
    if (goesLeft[idx]) {
        // Point goes to left sub-partition
        newLabel = newLabelBase;
        posInPartition = leftScan[idx];
    } else {
        // Point goes to right sub-partition (only if partition splits)
        // If partition doesn't split, all points stay in same partition
        if (partitionSplits) {
            newLabel = newLabelBase + 1;
            posInPartition = rightScan[idx];
        } else {
            // No split: use leftCount as offset for "right" points
            newLabel = newLabelBase;
            posInPartition = leftCountPerPartition[oldLabel] + rightScan[idx];
        }
    }
    
    int newPos = partitionStart[newLabel] + posInPartition;
    
    pxNew[newPos] = px[idx];
    pyNew[newPos] = py[idx];
    labelsNew[newPos] = newLabel;
}

// Extract the count of left-going points per partition (last value in each segment after inclusive scan)
__global__ void extractPartitionCountsKernel(int *labels, unsigned int *flags,
                                              int *leftScanInclusive, int *rightScanInclusive,
                                              int *leftCount, int *rightCount, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Check if this is the last element of a segment
    // Note: check idx < n-1 first to avoid out-of-bounds read of flags[n]
    bool isLastInSegment = (idx == n - 1) || (idx < n - 1 && flags[idx + 1] == 1);
    
    if (isLastInSegment) {
        int label = labels[idx];
        leftCount[label] = leftScanInclusive[idx];
        rightCount[label] = rightScanInclusive[idx];
    }
}

// CUDPP segmented exclusive scan for computing indices
void cudppSegmentedExclusiveScan(int *d_input, unsigned int *d_flags, int *d_output, int n) {
    initCUDPP();
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SEGMENTED_SCAN;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.options = CUDPP_OPTION_EXCLUSIVE | CUDPP_OPTION_FORWARD;
    
    CUDPPHandle plan;
    cudppPlan(cudppHandle, &plan, config, n, 1, 0);
    cudppSegmentedScan(plan, d_output, d_input, d_flags, n);
    cudppDestroyPlan(plan);
}

// CUDPP segmented inclusive scan for getting total counts per partition
void cudppSegmentedInclusiveScan(int *d_input, unsigned int *d_flags, int *d_output, int n) {
    initCUDPP();
    
    CUDPPConfiguration config;
    config.algorithm = CUDPP_SEGMENTED_SCAN;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_INT;
    config.options = CUDPP_OPTION_INCLUSIVE | CUDPP_OPTION_FORWARD;
    
    CUDPPHandle plan;
    cudppPlan(cudppHandle, &plan, config, n, 1, 0);
    cudppSegmentedScan(plan, d_output, d_input, d_flags, n);
    cudppDestroyPlan(plan);
}

// ============================================================================
// Main QuickHull function for GPU
// ============================================================================
extern "C" void gpuQuickHull(float *h_px, float *h_py, int n, 
                              float *result_x, float *result_y, int *M) {
    if (n <= 2) {
        for (int i = 0; i < n; i++) {
            result_x[i] = h_px[i];
            result_y[i] = h_py[i];
        }
        *M = n;
        return;
    }
    
    initCUDPP();

    // Device memory
    float *d_px, *d_py, *d_pxNew, *d_pyNew;
    int *d_labels, *d_labelsNew;
    float *d_distances;
    
    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMalloc(&d_pxNew, n * sizeof(float));
    cudaMalloc(&d_pyNew, n * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));
    cudaMalloc(&d_labelsNew, n * sizeof(int));
    cudaMalloc(&d_distances, n * sizeof(float));
    
    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Steps 5-6: Find min and max points
    float minX, maxX;
    findMinMaxWithCUDPP(d_px, n, &minX, &maxX);
    
    float minY = FLT_MAX, maxY = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (h_px[i] == minX && h_py[i] < minY) minY = h_py[i];
        if (h_px[i] == maxX && h_py[i] > maxY) maxY = h_py[i];
    }
    
    // Initialize ANS array
    int maxAnsSize = n + 2;
    float *h_ansX = new float[maxAnsSize];
    float *h_ansY = new float[maxAnsSize];
    
    h_ansX[0] = minX; h_ansY[0] = minY;
    h_ansX[1] = maxX; h_ansY[1] = maxY;
    h_ansX[2] = minX; h_ansY[2] = minY;
    int ansSize = 3;
    int numPartitions = 2;
    
    float *d_ansX, *d_ansY;
    cudaMalloc(&d_ansX, maxAnsSize * sizeof(float));
    cudaMalloc(&d_ansY, maxAnsSize * sizeof(float));
    cudaMemcpy(d_ansX, h_ansX, ansSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ansY, h_ansY, ansSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Steps 2-4: Initialize labels and sort by label    
    float *h_pxSorted = new float[n];
    float *h_pySorted = new float[n];
    int *h_labelsSorted = new int[n];
    int idx0 = 0, idx1 = n-1;
    
    for (int i = 0; i < n; i++) {
        float d = (maxX - minX) * (h_py[i] - minY) - (maxY - minY) * (h_px[i] - minX);
        if (d > 0) {
            h_pxSorted[idx0] = h_px[i];
            h_pySorted[idx0] = h_py[i];
            h_labelsSorted[idx0] = 0;
            idx0++;
        } else {
            h_pxSorted[idx1] = h_px[i];
            h_pySorted[idx1] = h_py[i];
            h_labelsSorted[idx1] = 1;
            idx1--;
        }
    }
    
    cudaMemcpy(d_px, h_pxSorted, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_pySorted, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labelsSorted, n * sizeof(int), cudaMemcpyHostToDevice);
    
    delete[] h_pxSorted;
    delete[] h_pySorted;
    delete[] h_labelsSorted;
    
    // Partition processing arrays
    float *d_maxDist;
    int *d_maxIdx;
    int *d_state, *d_statePrefix;
    int *d_goesLeft, *d_goesRight;
    int *d_leftScan, *d_rightScan;
    int *d_leftScanInc, *d_rightScanInc;
    int *d_leftCount, *d_rightCount, *d_partitionStart;
    
    // Arrays for segmented scan (steps 22-30)
    unsigned int *d_segmentFlags;
    float *d_scanResult;
    
    cudaMalloc(&d_maxDist, maxAnsSize * sizeof(float));
    cudaMalloc(&d_maxIdx, maxAnsSize * sizeof(int));
    cudaMalloc(&d_state, maxAnsSize * sizeof(int));
    cudaMalloc(&d_statePrefix, maxAnsSize * sizeof(int));
    cudaMalloc(&d_goesLeft, n * sizeof(int));
    cudaMalloc(&d_goesRight, n * sizeof(int));
    cudaMalloc(&d_leftScan, n * sizeof(int));
    cudaMalloc(&d_rightScan, n * sizeof(int));
    cudaMalloc(&d_leftScanInc, n * sizeof(int));
    cudaMalloc(&d_rightScanInc, n * sizeof(int));
    cudaMalloc(&d_leftCount, maxAnsSize * sizeof(int));
    cudaMalloc(&d_rightCount, maxAnsSize * sizeof(int));
    cudaMalloc(&d_partitionStart, maxAnsSize * sizeof(int));
    
    // Allocate segmented scan arrays
    cudaMalloc(&d_segmentFlags, n * sizeof(unsigned int));
    cudaMalloc(&d_scanResult, n * sizeof(float));
    
    int *h_state = new int[maxAnsSize];
    int *h_statePrefix = new int[maxAnsSize];
    int *h_maxIdx = new int[maxAnsSize];
    int *h_leftCount = new int[maxAnsSize];
    int *h_rightCount = new int[maxAnsSize];
    int *h_partitionStart = new int[maxAnsSize];
    
    int currentN = n;
    bool changed = true;
    int iteration = 0;

    printf("[DEBUG] Starting QuickHull main loop with n=%d, numPartitions=%d\n", currentN, numPartitions);
    fflush(stdout);

    // Main loop (steps 7-53)
    while (changed && currentN > 0) {
        changed = false;
        int numBlocks = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        printf("[DEBUG] Iteration %d: currentN=%d, numPartitions=%d, numBlocks=%d\n", 
               iteration, currentN, numPartitions, numBlocks);
        fflush(stdout);

        // Steps 8-16: Compute distances
        printf("[DEBUG]   Computing distances...\n"); fflush(stdout);
        int sharedMemSize = 2 * (BLOCK_SIZE + 2) * sizeof(float);
        computeDistancesKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_labels, d_ansX, d_ansY, ansSize, d_distances, currentN);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] computeDistancesKernel failed: %s\n", cudaGetErrorString(err));
            fflush(stdout);
        }
        printf("[DEBUG]   Distances computed.\n"); fflush(stdout);
        
        // Steps 17-20: Initialize arrays
        printf("[DEBUG]   Initializing state/maxIdx arrays...\n"); fflush(stdout);
        cudaMemset(d_state, 0, numPartitions * sizeof(int));
        
        // Initialize maxIdx to INT_MAX (so atomicMin can find the minimum index)
        std::vector<int> initMaxIdx(numPartitions, INT_MAX);
        cudaMemcpy(d_maxIdx, initMaxIdx.data(), numPartitions * sizeof(int), cudaMemcpyHostToDevice);
        
        // Steps 22-30: Find max distance point per partition using segmented scan
        // Create segment flags (1 at start of each partition)
        printf("[DEBUG]   Creating segment flags...\n"); fflush(stdout);
        createSegmentFlagsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_segmentFlags, currentN);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] createSegmentFlagsKernel failed: %s\n", cudaGetErrorString(err));
            fflush(stdout);
        }
        
        // Perform segmented max scan (negative distances won't win against positive ones)
        printf("[DEBUG]   Performing segmented max scan...\n"); fflush(stdout);
        cudppSegmentedMaxScan(d_distances, d_segmentFlags, d_scanResult, currentN);
        printf("[DEBUG]   Segmented max scan done.\n"); fflush(stdout);
        
        // Find max points from scan result (stores index atomically)
        printf("[DEBUG]   Finding max points from scan...\n"); fflush(stdout);
        findMaxPointFromScanKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels, d_distances, d_scanResult, d_segmentFlags,
            d_maxIdx, d_state, currentN);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] findMaxPointFromScanKernel failed: %s\n", cudaGetErrorString(err));
            fflush(stdout);
        }
        
        cudaMemcpy(h_state, d_state, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_maxIdx, d_maxIdx, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("[DEBUG]   State/maxIdx for partitions:\n"); fflush(stdout);
        for (int i = 0; i < numPartitions; i++) {
            printf("[DEBUG]     partition %d: state=%d, maxIdx=%d\n", i, h_state[i], h_maxIdx[i]);
        }
        fflush(stdout);

        for (int i = 0; i < numPartitions; i++) {
            if (h_state[i] == 1) {
                changed = true;
                break;
            }
        }
        printf("[DEBUG]   changed=%d\n", changed ? 1 : 0); fflush(stdout);
        if (!changed) break;
        
        // Steps 32-35: Compute prefix sum of state array
        printf("[DEBUG]   Computing state prefix sum...\n"); fflush(stdout);
        cudppExclusiveScan(d_state, d_statePrefix, numPartitions);
        cudaMemcpy(h_statePrefix, d_statePrefix, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
        printf("[DEBUG]   State prefix: ");
        for (int i = 0; i < numPartitions; i++) printf("%d ", h_statePrefix[i]);
        printf("\n"); fflush(stdout);
        
        // Steps 50-52 (executed before 36-47): Update ANS array
        // Note: We update ANS first because classifyPointsKernel reads the new
        // partition endpoints (including MAX points) from ANS. This is equivalent
        // to the pseudocode which uses a separate MAX[] array during classification.
        printf("[DEBUG]   Updating ANS array...\n"); fflush(stdout);
        float *h_ansXNew = new float[maxAnsSize];
        float *h_ansYNew = new float[maxAnsSize];
        int newAnsSize = 0;
        
        // Read coordinates of max points using their indices
        float *h_pxTemp = new float[currentN];
        float *h_pyTemp = new float[currentN];
        cudaMemcpy(h_pxTemp, d_px, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pyTemp, d_py, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < numPartitions; i++) {
            h_ansXNew[newAnsSize] = h_ansX[i];
            h_ansYNew[newAnsSize] = h_ansY[i];
            newAnsSize++;
            
            if (h_state[i] == 1) {
                int idx = h_maxIdx[i];
                h_ansXNew[newAnsSize] = h_pxTemp[idx];
                h_ansYNew[newAnsSize] = h_pyTemp[idx];
                newAnsSize++;
            }
        }
        h_ansXNew[newAnsSize] = h_ansX[numPartitions];
        h_ansYNew[newAnsSize] = h_ansY[numPartitions];
        newAnsSize++;
        
        delete[] h_pxTemp;
        delete[] h_pyTemp;
        
        for (int i = 0; i < newAnsSize; i++) {
            h_ansX[i] = h_ansXNew[i];
            h_ansY[i] = h_ansYNew[i];
        }
        ansSize = newAnsSize;
        int newNumPartitions = ansSize - 1;
        
        cudaMemcpy(d_ansX, h_ansX, ansSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ansY, h_ansY, ansSize * sizeof(float), cudaMemcpyHostToDevice);
        
        delete[] h_ansXNew;
        delete[] h_ansYNew;

        printf("[DEBUG]   ANS updated: ansSize=%d, newNumPartitions=%d\n", ansSize, newNumPartitions);
        fflush(stdout);

        // Steps 36-47: Compact and relabel using prefix scans (as per paper)
        // Now that ANS contains the new MAX points, we can classify points.
        // a) Classify points into left/right sub-partitions
        printf("[DEBUG]   Classifying points...\n"); fflush(stdout);
        classifyPointsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels, d_distances, d_statePrefix,
            d_ansX, d_ansY, d_goesLeft, d_goesRight, currentN);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] classifyPointsKernel failed: %s\n", cudaGetErrorString(err));
            fflush(stdout);
        }
        
        // Create segment flags for segmented scans
        printf("[DEBUG]   Creating partition flags...\n"); fflush(stdout);
        createPartitionFlagsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_segmentFlags, currentN);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] createPartitionFlagsKernel failed: %s\n", cudaGetErrorString(err));
            fflush(stdout);
        }
        
        // b) Use prefix scan to determine indices for left sub-partition
        printf("[DEBUG]   Segmented exclusive scan for left...\n"); fflush(stdout);
        cudppSegmentedExclusiveScan(d_goesLeft, d_segmentFlags, d_leftScan, currentN);
        
        // c) Use prefix scan to determine indices for right sub-partition  
        printf("[DEBUG]   Segmented exclusive scan for right...\n"); fflush(stdout);
        cudppSegmentedExclusiveScan(d_goesRight, d_segmentFlags, d_rightScan, currentN);
        
        // Get total counts per partition using inclusive scan
        printf("[DEBUG]   Segmented inclusive scans...\n"); fflush(stdout);
        cudppSegmentedInclusiveScan(d_goesLeft, d_segmentFlags, d_leftScanInc, currentN);
        cudppSegmentedInclusiveScan(d_goesRight, d_segmentFlags, d_rightScanInc, currentN);
        printf("[DEBUG]   Scans done.\n"); fflush(stdout);
        
        // Extract counts from the last element of each segment
        printf("[DEBUG]   Extracting partition counts...\n"); fflush(stdout);
        cudaMemset(d_leftCount, 0, numPartitions * sizeof(int));
        cudaMemset(d_rightCount, 0, numPartitions * sizeof(int));
        extractPartitionCountsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_segmentFlags, d_leftScanInc, d_rightScanInc,
            d_leftCount, d_rightCount, currentN);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] extractPartitionCountsKernel failed: %s\n", cudaGetErrorString(err));
            fflush(stdout);
        }
        
        cudaMemcpy(h_leftCount, d_leftCount, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rightCount, d_rightCount, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
        
        printf("[DEBUG]   Left/Right counts per partition:\n");
        for (int i = 0; i < numPartitions; i++) {
            printf("[DEBUG]     partition %d: left=%d, right=%d\n", i, h_leftCount[i], h_rightCount[i]);
        }
        fflush(stdout);

        // Compute partition start positions
        printf("[DEBUG]   Computing partition start positions...\n"); fflush(stdout);
        h_partitionStart[0] = 0;
        int newPartIdx = 0;
        for (int i = 0; i < numPartitions; i++) {
            if (h_state[i] == 1) {
                // Partition splits: left sub-partition first, then right
                h_partitionStart[newPartIdx + 1] = h_partitionStart[newPartIdx] + h_leftCount[i];
                newPartIdx++;
                h_partitionStart[newPartIdx + 1] = h_partitionStart[newPartIdx] + h_rightCount[i];
                newPartIdx++;
            } else {
                // Partition doesn't split: all positive points stay together
                h_partitionStart[newPartIdx + 1] = h_partitionStart[newPartIdx] + h_leftCount[i] + h_rightCount[i];
                newPartIdx++;
            }
        }
        
        int newN = h_partitionStart[newNumPartitions];
        
        printf("[DEBUG]   newN=%d, partition starts: ", newN);
        for (int i = 0; i <= newNumPartitions; i++) printf("%d ", h_partitionStart[i]);
        printf("\n"); fflush(stdout);

        if (newN > 0) {
            printf("[DEBUG]   Compacting points...\n"); fflush(stdout);
            cudaMemcpy(d_partitionStart, h_partitionStart, (newNumPartitions + 1) * sizeof(int), cudaMemcpyHostToDevice);
            
            compactWithScanKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px, d_py, d_labels, d_goesLeft, d_goesRight,
                d_leftScan, d_rightScan, d_leftCount, d_statePrefix,
                d_state, d_partitionStart, d_pxNew, d_pyNew, d_labelsNew, currentN);
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("[ERROR] compactWithScanKernel failed: %s\n", cudaGetErrorString(err));
                fflush(stdout);
            }
            printf("[DEBUG]   Compact done, swapping buffers.\n"); fflush(stdout);
            
            float *tmp; int *tmpi;
            tmp = d_px; d_px = d_pxNew; d_pxNew = tmp;
            tmp = d_py; d_py = d_pyNew; d_pyNew = tmp;
            tmpi = d_labels; d_labels = d_labelsNew; d_labelsNew = tmpi;
        }
        
        currentN = newN;
        numPartitions = newNumPartitions;
        iteration++;
        printf("[DEBUG] End of iteration %d: currentN=%d, numPartitions=%d\n\n", 
               iteration-1, currentN, numPartitions);
        fflush(stdout);
        
        // Safety check to prevent infinite loops
        if (iteration > 100) {
            printf("[ERROR] Too many iterations, breaking to prevent infinite loop!\n");
            fflush(stdout);
            break;
        }
    }
    
    printf("[DEBUG] Main loop ended. Final ansSize=%d\n", ansSize);
    fflush(stdout);
    
    // Extract hull points
    int hullSize = 0;
    for (int i = 0; i < ansSize - 1; i++) {
        result_x[hullSize] = h_ansX[i];
        result_y[hullSize] = h_ansY[i];
        hullSize++;
    }
    *M = hullSize;
    
    // Cleanup
    cudaFree(d_px); cudaFree(d_py);
    cudaFree(d_pxNew); cudaFree(d_pyNew);
    cudaFree(d_labels); cudaFree(d_labelsNew);
    cudaFree(d_distances);
    cudaFree(d_ansX); cudaFree(d_ansY);
    cudaFree(d_maxDist); cudaFree(d_maxIdx);
    cudaFree(d_state); cudaFree(d_statePrefix);
    cudaFree(d_goesLeft); cudaFree(d_goesRight);
    cudaFree(d_leftScan); cudaFree(d_rightScan);
    cudaFree(d_leftScanInc); cudaFree(d_rightScanInc);
    cudaFree(d_leftCount); cudaFree(d_rightCount);
    cudaFree(d_partitionStart);
    cudaFree(d_segmentFlags);
    cudaFree(d_scanResult);
    
    delete[] h_ansX; delete[] h_ansY;
    delete[] h_state; delete[] h_statePrefix;
    delete[] h_maxIdx;
    delete[] h_leftCount; delete[] h_rightCount;
    delete[] h_partitionStart;
}