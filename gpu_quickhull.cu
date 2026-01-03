#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <cub/device/device_segmented_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include "utils.h"


#define BLOCK_SIZE 512

// Custom sum operator for CUB segmented scans
struct SumOp {
    __host__ __device__ __forceinline__
    int operator()(const int &a, const int &b) const {
        return a + b;
    }
};


// Steps 5-6
void findMinMaxWithCUB(float *d_px, int n, float *minX, float *maxX) {
    // CUB DeviceReduce for min
    float *d_minX;
    cudaMalloc(&d_minX, sizeof(float));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_px, d_minX, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_px, d_minX, n);
    cudaMemcpy(minX, d_minX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_minX);

    // CUB DeviceReduce for max
    float *d_maxX;
    cudaMalloc(&d_maxX, sizeof(float));
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_px, d_maxX, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_px, d_maxX, n);
    cudaMemcpy(maxX, d_maxX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);
    cudaFree(d_maxX);
}

void cubExclusiveScan(int *d_input, int *d_output, int n) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaFree(d_temp_storage);
}

// Steps 8-16
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

// Steps 22-30
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

// After segmented max reduce, find points that have the max distance in their partition
// maxDistPerPartition: array of size numPartitions, contains max distance for each partition
__global__ void findMaxPointFromReduceKernel(float *px, float *py, int *labels,
                                            float *distances, float *maxDistPerPartition,
                                            int *maxIdx, int *state,
                                            int n, int numPartitions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float d = distances[idx];
    int label = labels[idx];

    // Get the max distance for this partition from the reduced output
    float maxInPartition = maxDistPerPartition[label];

    // Set state for this partition (any thread in the partition can do this, result is same)
    // We use atomicMax to ensure at least one thread sets it
    if (maxInPartition > 0) {
        state[label] = 1;
    }

    // If this point has positive distance and matches the partition maximum,
    // try to publish its index as the partition max index.
    if (d > 0 && d == maxInPartition) {
        atomicMin(&maxIdx[label], idx);
    }
}

// Output: d_output is sized for num_partitions (not n), stores max distance per partition
void cubSegmentedMaxReduce(float *d_input, unsigned int *d_flags, float *d_output, int n, int num_partitions) {
    // d_flags: 1 at segment start, 0 elsewhere
    // Convert flags to segment offsets
    int num_segments = 0;
    std::vector<int> h_offsets;
    h_offsets.reserve(n+1);
    cudaDeviceSynchronize();
    unsigned int* h_flags = new unsigned int[n];
    cudaMemcpy(h_flags, d_flags, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        if (h_flags[i] == 1) {
            h_offsets.push_back(i);
            num_segments++;
        }
    }
    h_offsets.push_back(n);
    int* d_offsets;
    cudaMalloc(&d_offsets, (num_segments+1) * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets.data(), (num_segments+1) * sizeof(int), cudaMemcpyHostToDevice);
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output, num_segments, d_offsets, d_offsets+1);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSegmentedReduce::Max(d_temp_storage, temp_storage_bytes, d_input, d_output, num_segments, d_offsets, d_offsets+1);
    cudaFree(d_temp_storage);
    cudaFree(d_offsets);
    delete[] h_flags;
}

// Steps 36-47
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

// segmented exclusive scan for computing indices
void cubSegmentedExclusiveScan(int *d_input, unsigned int *d_flags, int *d_output, int n) {
    // Convert flags to segment offsets
    int num_segments = 0;
    std::vector<int> h_offsets;
    h_offsets.reserve(n+1);
    unsigned int* h_flags = new unsigned int[n];
    cudaMemcpy(h_flags, d_flags, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            if (h_flags[i] == 1) {
                h_offsets.push_back(i);
                num_segments++;
            }
        }
        h_offsets.push_back(n);
        int* d_offsets;
        cudaMalloc(&d_offsets, (num_segments+1) * sizeof(int));
        cudaMemcpy(d_offsets, h_offsets.data(), (num_segments+1) * sizeof(int), cudaMemcpyHostToDevice);
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        // CUB requires separate begin/end offset iterators, a scan operator, and initial value
        SumOp sum_op;
        cub::DeviceSegmentedScan::ExclusiveSegmentedScan(d_temp_storage, temp_storage_bytes, d_input, d_output, d_offsets, d_offsets + 1, (int64_t)num_segments, sum_op, 0);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedScan::ExclusiveSegmentedScan(d_temp_storage, temp_storage_bytes, d_input, d_output, d_offsets, d_offsets + 1, (int64_t)num_segments, sum_op, 0);
        cudaFree(d_temp_storage);
        cudaFree(d_offsets);
        delete[] h_flags;
}

// segmented inclusive scan for getting total counts per partition
void cubSegmentedInclusiveScan(int *d_input, unsigned int *d_flags, int *d_output, int n) {
    // Convert flags to segment offsets
    int num_segments = 0;
    std::vector<int> h_offsets;
    h_offsets.reserve(n+1);
    unsigned int* h_flags = new unsigned int[n];
    cudaMemcpy(h_flags, d_flags, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            if (h_flags[i] == 1) {
                h_offsets.push_back(i);
                num_segments++;
            }
        }
        h_offsets.push_back(n);
        int* d_offsets;
        cudaMalloc(&d_offsets, (num_segments+1) * sizeof(int));
        cudaMemcpy(d_offsets, h_offsets.data(), (num_segments+1) * sizeof(int), cudaMemcpyHostToDevice);
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        // CUB requires separate begin/end offset iterators and a scan operator
        SumOp sum_op;
        cub::DeviceSegmentedScan::InclusiveSegmentedScan(d_temp_storage, temp_storage_bytes, d_input, d_output, d_offsets, d_offsets + 1, (int64_t)num_segments, sum_op);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSegmentedScan::InclusiveSegmentedScan(d_temp_storage, temp_storage_bytes, d_input, d_output, d_offsets, d_offsets + 1, (int64_t)num_segments, sum_op);
        cudaFree(d_temp_storage);
        cudaFree(d_offsets);
        delete[] h_flags;
}

// Main QuickHull Function
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

    // Steps 5-6
    float minX, maxX;
    findMinMaxWithCUB(d_px, n, &minX, &maxX);

    float minY = FLT_MAX, maxY = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (h_px[i] == minX && h_py[i] < minY) minY = h_py[i];
        if (h_px[i] == maxX && h_py[i] > maxY) maxY = h_py[i];
    }
    
    printf("DEBUG: Initial extreme points: minX=%.3f,minY=%.3f, maxX=%.3f,maxY=%.3f\n", minX, minY, maxX, maxY);

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

    // Steps 2-4
    float *h_pxSorted = new float[n];
    float *h_pySorted = new float[n];
    int *h_labelsSorted = new int[n];
    int idx0 = 0, idx1 = n-1;
    
    int positiveCount = 0, negativeCount = 0;
    for (int i = 0; i < n; i++) {
        float d = (maxX - minX) * (h_py[i] - minY) - (maxY - minY) * (h_px[i] - minX);
        if (d > 0) {
            h_pxSorted[idx0] = h_px[i];
            h_pySorted[idx0] = h_py[i];
            h_labelsSorted[idx0] = 0;
            idx0++;
            positiveCount++;
        } else {
            h_pxSorted[idx1] = h_px[i];
            h_pySorted[idx1] = h_py[i];
            h_labelsSorted[idx1] = 1;
            idx1--;
            negativeCount++;
        }
    }
    printf("DEBUG: Initial partitioning: positive=%d, negative=%d\n", positiveCount, negativeCount);

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

    // steps 22-30
    unsigned int *d_segmentFlags;
    float *d_maxDistPerPartition;

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
    cudaMalloc(&d_maxDistPerPartition, maxAnsSize * sizeof(float));  // One max per partition

    int *h_state = new int[maxAnsSize];
    int *h_statePrefix = new int[maxAnsSize];
    int *h_maxIdx = new int[maxAnsSize];
    int *h_leftCount = new int[maxAnsSize];
    int *h_rightCount = new int[maxAnsSize];
    int *h_partitionStart = new int[maxAnsSize];

    int currentN = n;
    bool changed = true;
    int iteration = 0;

    // steps 7-53
    while (changed && currentN > 0) {
        printf("DEBUG: Iteration %d: currentN=%d, numPartitions=%d\n", iteration++, currentN, numPartitions);
        changed = false;
        int numBlocks = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Steps 8-16
        int sharedMemSize = 2 * (BLOCK_SIZE + 2) * sizeof(float);
        computeDistancesKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_labels, d_ansX, d_ansY, ansSize, d_distances, currentN);
        cudaDeviceSynchronize();

        // Debug: Check some distance values
        float *h_distances = new float[currentN];
        cudaMemcpy(h_distances, d_distances, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        printf("DEBUG: All distances: ");
        for (int i = 0; i < currentN; i++) {
            printf("%.3f ", h_distances[i]);
        }
        printf("\n");
        
        // Count positive distances
        int posDistCount = 0;
        float maxDist = -FLT_MAX;
        for (int i = 0; i < currentN; i++) {
            if (h_distances[i] > 0) posDistCount++;
            if (h_distances[i] > maxDist) maxDist = h_distances[i];
        }
        printf("DEBUG: Positive distances: %d/%d, max distance: %.6f\n", posDistCount, currentN, maxDist);
        delete[] h_distances;

        // Steps 17-20
        cudaMemset(d_state, 0, numPartitions * sizeof(int));

        // Initialize maxIdx to INT_MAX (so atomicMin can find the minimum index)
        std::vector<int> initMaxIdx(numPartitions, INT_MAX);
        cudaMemcpy(d_maxIdx, initMaxIdx.data(), numPartitions * sizeof(int), cudaMemcpyHostToDevice);

        // Steps 22-30
        // Find max distance point per partition using segmented reduce
        // Create segment flags (1 at start of each partition)
        createSegmentFlagsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_segmentFlags, currentN);
        cudaDeviceSynchronize();

        // Perform segmented max reduce (outputs one max per partition)
        cubSegmentedMaxReduce(d_distances, d_segmentFlags, d_maxDistPerPartition, currentN, numPartitions);

        // Debug: Check max distance per partition
        float *h_maxDistPerPartition = new float[numPartitions];
        cudaMemcpy(h_maxDistPerPartition, d_maxDistPerPartition, numPartitions * sizeof(float), cudaMemcpyDeviceToHost);
        printf("DEBUG: Max distance per partition: ");
        for (int i = 0; i < numPartitions; i++) {
            printf("P%d:%.3f ", i, h_maxDistPerPartition[i]);
        }
        printf("\n");
        delete[] h_maxDistPerPartition;

        // Find max points from reduce result (stores index atomically)
        findMaxPointFromReduceKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels, d_distances, d_maxDistPerPartition,
            d_maxIdx, d_state, currentN, numPartitions);
        cudaDeviceSynchronize();

        cudaMemcpy(h_state, d_state, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_maxIdx, d_maxIdx, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);

        printf("DEBUG: Partition states: ");
        for (int i = 0; i < numPartitions; i++) {
            printf("P%d:state=%d,maxIdx=%d ", i, h_state[i], h_maxIdx[i]);
        }
        printf("\n");

        for (int i = 0; i < numPartitions; i++) {
            if (h_state[i] == 1) {
                changed = true;
                break;
            }
        }
        if (!changed) break;

        // Steps 32-35
        cubExclusiveScan(d_state, d_statePrefix, numPartitions);
        cudaMemcpy(h_statePrefix, d_statePrefix, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);

        // Steps 50-52
        // Note: We update ANS first because classifyPointsKernel reads the new
        // partition endpoints (including MAX points) from ANS. This is equivalent
        // to the pseudocode which uses a separate MAX[] array during classification.
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

        // Steps 36-47
        // Now that ANS contains the new MAX points, we can classify points.
        // a) Classify points into left/right sub-partitions
        classifyPointsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels, d_distances, d_statePrefix,
            d_ansX, d_ansY, d_goesLeft, d_goesRight, currentN);
        cudaDeviceSynchronize();

        // Create segment flags for segmented scans
        createPartitionFlagsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_segmentFlags, currentN);
        cudaDeviceSynchronize();

        // b) Use prefix scan to determine indices for left sub-partition
        cubSegmentedExclusiveScan(d_goesLeft, d_segmentFlags, d_leftScan, currentN);

        // c) Use prefix scan to determine indices for right sub-partition
        cubSegmentedExclusiveScan(d_goesRight, d_segmentFlags, d_rightScan, currentN);

        // Get total counts per partition using inclusive scan
        cubSegmentedInclusiveScan(d_goesLeft, d_segmentFlags, d_leftScanInc, currentN);
        cubSegmentedInclusiveScan(d_goesRight, d_segmentFlags, d_rightScanInc, currentN);

        // Extract counts from the last element of each segment
        cudaMemset(d_leftCount, 0, numPartitions * sizeof(int));
        cudaMemset(d_rightCount, 0, numPartitions * sizeof(int));
        extractPartitionCountsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_segmentFlags, d_leftScanInc, d_rightScanInc,
            d_leftCount, d_rightCount, currentN);
        cudaDeviceSynchronize();

        cudaMemcpy(h_leftCount, d_leftCount, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rightCount, d_rightCount, numPartitions * sizeof(int), cudaMemcpyDeviceToHost);

        // Debug: Check partition counts
        printf("DEBUG: Partition counts: ");
        for (int i = 0; i < numPartitions; i++) {
            printf("P%d:left=%d,right=%d ", i, h_leftCount[i], h_rightCount[i]);
        }
        printf("\n");

        // Compute partition start positions
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
        printf("DEBUG: After partitioning: newN=%d, newNumPartitions=%d\n", newN, newNumPartitions);

        if (newN > 0) {
            cudaMemcpy(d_partitionStart, h_partitionStart, (newNumPartitions + 1) * sizeof(int), cudaMemcpyHostToDevice);

            compactWithScanKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px, d_py, d_labels, d_goesLeft, d_goesRight,
                d_leftScan, d_rightScan, d_leftCount, d_statePrefix,
                d_state, d_partitionStart, d_pxNew, d_pyNew, d_labelsNew, currentN);
            cudaDeviceSynchronize();

            float *tmp; int *tmpi;
            tmp = d_px; d_px = d_pxNew; d_pxNew = tmp;
            tmp = d_py; d_py = d_pyNew; d_pyNew = tmp;
            tmpi = d_labels; d_labels = d_labelsNew; d_labelsNew = tmpi;
        }

        currentN = newN;
        numPartitions = newNumPartitions;
    }

    // Extract hull points
    int hullSize = 0;
    printf("DEBUG: Final ansSize=%d\n", ansSize);
    for (int i = 0; i < ansSize - 1; i++) {
        printf("DEBUG: Hull point %d: (%.3f, %.3f)\n", i, h_ansX[i], h_ansY[i]);
        result_x[hullSize] = h_ansX[i];
        result_y[hullSize] = h_ansY[i];
        hullSize++;
    }
    *M = hullSize;
    printf("DEBUG: Final hull size: %d\n", hullSize);

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
    cudaFree(d_maxDistPerPartition);

    delete[] h_ansX; delete[] h_ansY;
    delete[] h_state; delete[] h_statePrefix;
    delete[] h_maxIdx;
    delete[] h_leftCount; delete[] h_rightCount;
    delete[] h_partitionStart;
}