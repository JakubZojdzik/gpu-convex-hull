#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_histogram.cuh>
#include "utils.h"

#define BLOCK_SIZE 512

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

__global__ void buildPointArray(const float *px,
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

void findMinMaxX_CUB(const float *d_px,
                     const float *d_py,
                     int n,
                     MinMaxPoint &h_min,
                     MinMaxPoint &h_max)
{
    // Build point array
    MinMaxPoint *d_points;
    cudaMalloc(&d_points, n * sizeof(MinMaxPoint));

    int grid  = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    buildPointArray<<<grid, BLOCK_SIZE>>>(d_px, d_py, d_points, n);

    // Output buffers
    MinMaxPoint *d_min, *d_max;
    cudaMalloc(&d_min, sizeof(MinMaxPoint));
    cudaMalloc(&d_max, sizeof(MinMaxPoint));

    // Temporary storage
    void *d_temp = nullptr;
    size_t temp_bytes = 0;

    // Query temp size (min)
    cub::DeviceReduce::Reduce(
        d_temp, temp_bytes,
        d_points, d_min,
        n,
        MinXOp(),
        MinMaxPoint{FLT_MAX, FLT_MAX, -1}
    );

    cudaMalloc(&d_temp, temp_bytes);

    // Run min reduction
    cub::DeviceReduce::Reduce(
        d_temp, temp_bytes,
        d_points, d_min,
        n,
        MinXOp(),
        MinMaxPoint{FLT_MAX, FLT_MAX, -1}
    );

    // Run max reduction (reuse temp storage)
    cub::DeviceReduce::Reduce(
        d_temp, temp_bytes,
        d_points, d_max,
        n,
        MaxXOp(),
        MinMaxPoint{-FLT_MAX, -FLT_MAX, -1}
    );

    // Copy back
    cudaMemcpy(&h_min, d_min, sizeof(MinMaxPoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(MinMaxPoint), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_temp);
    cudaFree(d_points);
    cudaFree(d_min);
    cudaFree(d_max);
}

// Pair of distance and original point index
struct DistIdxPair {
    float dist;
    int   idx;
};

// Reduction operator: returns pair with larger distance
struct MaxDistOp {
    __host__ __device__
    DistIdxPair operator()(const DistIdxPair &a, const DistIdxPair &b) const {
        return (a.dist >= b.dist) ? a : b;
    }
};

// Kernel to build DistIdxPair array from distances
__global__ void buildDistIdxArray(const float *distances, DistIdxPair *pairs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pairs[i].dist = distances[i];
        pairs[i].idx = i;
    }
}

// Kernel to find segment offsets from sorted labels
// For each segment i, offsets[i] = first index where label == i
// offsets[numSegments] = n (sentinel)
__global__ void findSegmentOffsetsKernel(const int *labels, int *offsets, int numSegments, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First thread sets sentinel and the offset for the first label in the array
    if (i == 0) {
        offsets[numSegments] = n;  // Sentinel
        // The first point's label determines where that segment starts (at index 0)
        offsets[labels[0]] = 0;
    }
    
    // Each thread checks if there's a segment boundary at position i+1
    if (i < n - 1) {
        if (labels[i] != labels[i + 1]) {
            // Boundary between segment labels[i] and labels[i+1]
            offsets[labels[i + 1]] = i + 1;
        }
    }
}


// ============================================================================
// Segmented max distance reduction using CUB
// Finds the max distance point for each partition (segment)
// Points must be SORTED BY LABEL for this to work correctly
// ============================================================================
void segmentedMaxDistReduce(
    const float *d_distances,
    const int *d_labels,
    int *d_segmentOffsets,  // Output: segment offsets [numSegments + 1]
    DistIdxPair *d_maxPerSegment,  // Output: max dist-idx pair per segment
    int n,
    int numSegments)
{
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Build segment offsets from sorted labels
    findSegmentOffsetsKernel<<<numBlocks, BLOCK_SIZE>>>(
        d_labels, d_segmentOffsets, numSegments, n);
    cudaDeviceSynchronize();

    // copy back to host
    int h_offsets[numSegments + 1];
    cudaMemcpy(h_offsets, d_segmentOffsets, (numSegments + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    // fill in any unset offsets
    for (int i = numSegments-1; i >= 1; i--) {
        if (h_offsets[i] == -1) {
            h_offsets[i] = h_offsets[i + 1];
        }
    }
    cudaMemcpy(d_segmentOffsets, h_offsets, (numSegments + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    // Build DistIdxPair array
    DistIdxPair *d_pairs;
    cudaMalloc(&d_pairs, n * sizeof(DistIdxPair));
    buildDistIdxArray<<<numBlocks, BLOCK_SIZE>>>(d_distances, d_pairs, n);
    cudaDeviceSynchronize();
    
    // Use CUB segmented reduce to find max per segment
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    
    DistIdxPair identity{0.0f, -1};
    
    // Query temp storage size
    cub::DeviceSegmentedReduce::Reduce(
        d_temp, temp_bytes,
        d_pairs, d_maxPerSegment,
        numSegments,
        d_segmentOffsets, d_segmentOffsets + 1,
        MaxDistOp(), identity);
    
    cudaMalloc(&d_temp, temp_bytes);
    
    // Run segmented reduce
    cub::DeviceSegmentedReduce::Reduce(
        d_temp, temp_bytes,
        d_pairs, d_maxPerSegment,
        numSegments,
        d_segmentOffsets, d_segmentOffsets + 1,
        MaxDistOp(), identity);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_temp);
    cudaFree(d_pairs);
}


// Compute distances for all points at once using labels
// Each point has a label indicating which partition it belongs to
// The partition's line segment is from ans[label] to ans[label+1]
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
    if (tid < ansRange) {
        if ((minLabel + tid) >= ansSize) {
            // impossible
           // printf("Error: ans index out of range in computeDistancesKernel\n");
            return;
        }
        sAnsX[tid] = ansX[minLabel + tid];
        sAnsY[tid] = ansY[minLabel + tid];
    }
    __syncthreads();

    if (idx >= n) return;

    int label = labels[idx];
    int localLabel = label - minLabel;

    // Get line endpoints from shared memory
    float lx = sAnsX[localLabel];
    float ly = sAnsY[localLabel];
    float rx = sAnsX[localLabel + 1];
    float ry = sAnsY[localLabel + 1];

    float curX = px[idx];
    float curY = py[idx];

    // Compute distance (cross product)
    float d = (rx - lx) * (curY - ly) - (ry - ly) * (curX - lx);
    distances[idx] = d;
}

// CUB exclusive scan wrapper
void cubExclusiveScanInt(int *d_input, int *d_output, int n) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaFree(d_temp_storage);
}

// Kernel to determine on which side of new max point each point lies
// Calculates state of partitions - 1 if partition splits, 0 if not
__global__ void determineSide(float *px, float *py, int *labels,
                                    float *ansX, float *ansY, int *state, DistIdxPair *maxIdxPerLabel,
                                    int *goesLeft, int *goesRight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = labels[idx];
    int maxIdx = maxIdxPerLabel[label].idx;
    
    if (maxIdx < 0) {
        // No max point found for this partition, point is eliminated
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        state[label] = 0;
        return;
    }

    if (maxIdx == idx) {
        // This point is the max point itself, it will be kept
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        state[label] = 1;
        return;
    }

    // Get L, M, R points
    float lx = ansX[label];
    float ly = ansY[label];
    float rx = ansX[label + 1];
    float ry = ansY[label + 1];
    float mx = px[maxIdx];
    float my = py[maxIdx];
    float curX = px[idx];
    float curY = py[idx];
    
    // Distance from L->M line (positive = left of line)
    float distLM = (mx - lx) * (curY - ly) - (my - ly) * (curX - lx);
    // Distance from M->R line (positive = left of line)
    float distMR = (rx - mx) * (curY - my) - (ry - my) * (curX - mx);
    
    // Points on the LEFT of L->M go to the left partition (label stays same)
    // Points on the LEFT of M->R go to the right partition (label increases by 1)
    if (distLM > 0) {
        goesLeft[idx] = 1;
        goesRight[idx] = 0;
    } else if (distMR > 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 1;
    }
}

// Kernel to compute new labels for points that survive
// Points are kept if goesLeft[i] || goesRight[i]
// New label = label[i] + statePrefixSum[label[i]] + goesRight[i]
// Non-kept points get label = newNumLabels (so histogram can count them separately)
__global__ void computeNewLabelsKernel(int *labels, int *statePrefixSum,
                                        int *goesLeft, int *goesRight,
                                        int *newLabels, int *keepFlags, 
                                        int n, int newNumLabels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int keep = goesLeft[idx] | goesRight[idx];
    keepFlags[idx] = keep;
    
    if (keep) {
        int label = labels[idx];
        newLabels[idx] = label + statePrefixSum[label] + goesRight[idx];
    } else {
        newLabels[idx] = newNumLabels;  // Will be removed, counted in separate histogram bin
    }
}

// CUB-based histogram for counting label occurrences
// Much more efficient than atomicAdd-based kernel
// newLabels should have values in [0, newNumLabels-1] for kept points,
// and newNumLabels for non-kept points (which will be counted but ignored)
void countPerLabelCUB(int *d_newLabels, int *d_labelCounts, int n, int newNumLabels) {
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    
    // HistogramEven counts values in [lower_level, upper_level) with num_levels-1 bins
    // We want bins for labels 0, 1, ..., newNumLabels-1, plus one extra bin for discarded points
    int num_levels = newNumLabels + 2;  // Creates newNumLabels+1 bins
    int lower_level = 0;
    int upper_level = newNumLabels + 1;
    
    // Query temp storage size
    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        d_newLabels, d_labelCounts,
        num_levels, lower_level, upper_level,
        n);
    
    cudaMalloc(&d_temp, temp_bytes);
    
    // Run histogram
    cub::DeviceHistogram::HistogramEven(
        d_temp, temp_bytes,
        d_newLabels, d_labelCounts,
        num_levels, lower_level, upper_level,
        n);
    
    cudaFree(d_temp);
}

// Kernel to compute local index within each label group
__global__ void computeLocalScatterIdx(int *newLabels, int *keepFlags,
                                        int *labelOffsets, int *scatterIdx, 
                                        int *labelCounters, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (keepFlags[idx]) {
        int newLabel = newLabels[idx];
        // Get a unique position within this label's group
        int localIdx = atomicAdd(&labelCounters[newLabel], 1);
        scatterIdx[idx] = labelOffsets[newLabel] + localIdx;
    }
}

// Kernel to compact points and labels based on scatter indices
__global__ void compactPointsKernel(float *pxIn, float *pyIn, int *labelsIn,
                                     float *pxOut, float *pyOut, int *labelsOut,
                                     int *keepFlags, int *scatterIdx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (keepFlags[idx]) {
        int outIdx = scatterIdx[idx];
        pxOut[outIdx] = pxIn[idx];
        pyOut[outIdx] = pyIn[idx];
        labelsOut[outIdx] = labelsIn[idx];
    }
}


// ============================================================================
// QuickHull for one side: finds hull points between leftPt and rightPt
// Points are assumed to be on the LEFT side of the directed edge leftPt->rightPt
// Returns hull points in order from leftPt to rightPt (exclusive of endpoints)
// Uses label-based approach to process all partitions at once
// ============================================================================
void gpuQuickHullOneSide(float *h_px, float *h_py, int n,
                          float leftX, float leftY, float rightX, float rightY,
                          std::vector<Point> &hullPoints) {
    if (n == 0) return;

    // Allocate device memory
    float *d_px, *d_py;
    float *d_distances;
    int *d_labels;
    float *d_ansX, *d_ansY;
    int *d_state;
    int *d_goesLeft, *d_goesRight;

    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMalloc(&d_distances, n * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));

    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize all labels to 0 (single partition)
    cudaMemset(d_labels, 0, n * sizeof(int));

    int maxAnsSize = n + 1;  // Upper bound on ANS size

    float *h_ansX, *h_ansY;
    h_ansX = (float*)malloc(maxAnsSize * sizeof(float));
    h_ansY = (float*)malloc(maxAnsSize * sizeof(float));
    h_ansX[0] = leftX;
    h_ansY[0] = leftY;
    h_ansX[1] = rightX;
    h_ansY[1] = rightY;
    
    int currentN = n;
    int numLabels = 1;  // Start with 1 partition (label 0)
    int ansSize = 2;
    
    // Allocate ANS arrays on device (will grow as needed)
    cudaMalloc(&d_ansX, maxAnsSize * sizeof(float));
    cudaMalloc(&d_ansY, maxAnsSize * sizeof(float));

    while (true) {
        // Copy current ANS to device
        cudaMemcpy(d_ansX, h_ansX, ansSize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ansY, h_ansY, ansSize * sizeof(float), cudaMemcpyHostToDevice);

        int numBlocks = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Compute distances for ALL points at once using labels
        size_t sharedMemSize = 2 * (BLOCK_SIZE + 2) * sizeof(float);
        computeDistancesKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_labels,
            d_ansX, d_ansY, ansSize,
            d_distances, currentN);
        cudaDeviceSynchronize();

        // Find max distance point for each partition using CUB segmented reduce
        // This follows the paper's methodology: since points are sorted by label,
        // we can use segmented operations efficiently
        int *d_segmentOffsets;
        DistIdxPair *d_maxPerSegment;
        cudaMalloc(&d_segmentOffsets, (numLabels + 1) * sizeof(int));
        cudaMemset(d_segmentOffsets, -1, (numLabels + 1) * sizeof(int));
        cudaMalloc(&d_maxPerSegment, numLabels * sizeof(DistIdxPair));
        
        segmentedMaxDistReduce(d_distances, d_labels, d_segmentOffsets, 
                               d_maxPerSegment, currentN, numLabels);
        cudaDeviceSynchronize();
        
        std::vector<DistIdxPair> h_maxPerSegment(numLabels);
        cudaMemcpy(h_maxPerSegment.data(), d_maxPerSegment, numLabels * sizeof(DistIdxPair), cudaMemcpyDeviceToHost);


        // determine where do points go (left/right of max point) and partition state
        cudaMalloc(&d_state, numLabels * sizeof(int));
        cudaMalloc(&d_goesLeft, currentN * sizeof(int));
        cudaMalloc(&d_goesRight, currentN * sizeof(int));
        cudaMemset(d_state, 0, numLabels * sizeof(int));
        cudaMemset(d_goesLeft, 0, currentN * sizeof(int));
        cudaMemset(d_goesRight, 0, currentN * sizeof(int));
        determineSide<<<numBlocks, BLOCK_SIZE>>>(d_px, d_py, d_labels, d_ansX, d_ansY, d_state, d_maxPerSegment, d_goesLeft, d_goesRight, currentN);
        cudaDeviceSynchronize();

        // print state for debugging
        std::vector<int> h_state(numLabels);
        cudaMemcpy(h_state.data(), d_state, numLabels * sizeof(int), cudaMemcpyDeviceToHost);

        // count state prefix sum
        int *d_statePrefixSum;
        cudaMalloc(&d_statePrefixSum, numLabels * sizeof(int));
        cubExclusiveScanInt(d_state, d_statePrefixSum, numLabels);
        cudaDeviceSynchronize();

        // print state prefix sum for debugging
        std::vector<int> h_statePrefixSum(numLabels);
        cudaMemcpy(h_statePrefixSum.data(), d_statePrefixSum, numLabels * sizeof(int), cudaMemcpyDeviceToHost);

        int numNewHullPoints = h_state[numLabels - 1] + h_statePrefixSum[numLabels - 1];
        // Calculate number of new labels (after inserting max points)
        int newNumLabels = numLabels + numNewHullPoints;
        
        // Compute new labels and keep flags for surviving points
        int *d_newLabels, *d_keepFlags, *d_scatterIdx;
        cudaMalloc(&d_newLabels, currentN * sizeof(int));
        cudaMalloc(&d_keepFlags, currentN * sizeof(int));
        cudaMalloc(&d_scatterIdx, currentN * sizeof(int));
        cudaMemset(d_keepFlags, 0, currentN * sizeof(int));
        
        computeNewLabelsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_statePrefixSum,
            d_goesLeft, d_goesRight,
            d_newLabels, d_keepFlags, currentN, newNumLabels);
        cudaDeviceSynchronize();
        
        // Count how many kept points go to each new label using CUB histogram
        // Allocate newNumLabels+1 to hold the extra bin for discarded points
        int *d_labelCounts, *d_labelOffsets, *d_labelCounters;
        cudaMalloc(&d_labelCounts, (newNumLabels + 1) * sizeof(int));
        cudaMalloc(&d_labelOffsets, newNumLabels * sizeof(int));
        cudaMalloc(&d_labelCounters, newNumLabels * sizeof(int));
        cudaMemset(d_labelCounters, 0, newNumLabels * sizeof(int));
        
        countPerLabelCUB(d_newLabels, d_labelCounts, currentN, newNumLabels);
        cudaDeviceSynchronize();
        
        // Compute prefix sum of label counts to get starting offset for each label
        // Only scan the first newNumLabels bins (ignore the discard bin)
        cubExclusiveScanInt(d_labelCounts, d_labelOffsets, newNumLabels);
        cudaDeviceSynchronize();
        
        // Compute scatter indices that maintain label order
        computeLocalScatterIdx<<<numBlocks, BLOCK_SIZE>>>(
            d_newLabels, d_keepFlags, d_labelOffsets, d_scatterIdx, d_labelCounters, currentN);
        cudaDeviceSynchronize();

        std::vector<int> h_labelCounts(newNumLabels);
        cudaMemcpy(h_labelCounts.data(), d_labelCounts, newNumLabels * sizeof(int), cudaMemcpyDeviceToHost);
        int newN = 0;
        for (int i = 0; i < newNumLabels; i++) {
            newN += h_labelCounts[i];
        }
        
        // If no new hull points found, we're done
        if (numNewHullPoints == 0) {
            cudaFree(d_segmentOffsets);
            cudaFree(d_maxPerSegment);
            cudaFree(d_state);
            cudaFree(d_goesLeft);
            cudaFree(d_goesRight);
            cudaFree(d_statePrefixSum);
            cudaFree(d_newLabels);
            cudaFree(d_keepFlags);
            cudaFree(d_scatterIdx);
            cudaFree(d_labelCounts);
            cudaFree(d_labelOffsets);
            cudaFree(d_labelCounters);
            break;
        }
        
        // Copy point coordinates to host to get max point coords
        std::vector<float> h_px_temp(currentN), h_py_temp(currentN);
        cudaMemcpy(h_px_temp.data(), d_px, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_py_temp.data(), d_py, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Build new ANS by inserting max points in correct positions
        // For each partition that splits, insert the max point between L and R
        float* h_newAnsX = (float*)malloc((newNumLabels + 1) * sizeof(float));
        float* h_newAnsY = (float*)malloc((newNumLabels + 1) * sizeof(float));
        int newAnsSize = 0;
        for (int i = 0; i < numLabels; i++) {
            // Add the left endpoint of partition i
            h_newAnsX[newAnsSize] = h_ansX[i];
            h_newAnsY[newAnsSize] = h_ansY[i];
            newAnsSize++;
            
            // If partition i splits, add its max point
            if (h_state[i] == 1 && h_maxPerSegment[i].idx >= 0) {
                int maxIdx = h_maxPerSegment[i].idx;
                h_newAnsX[newAnsSize] = h_px_temp[maxIdx];
                h_newAnsY[newAnsSize] = h_py_temp[maxIdx];
                newAnsSize++;
            }
        }
        // Add the final endpoint
        h_newAnsX[newAnsSize] = h_ansX[numLabels];
        h_newAnsY[newAnsSize] = h_ansY[numLabels];
        newAnsSize++;
        free(h_ansX);
        free(h_ansY);
        h_ansX = h_newAnsX;
        h_ansY = h_newAnsY;

        if (newN == 0) {
            cudaFree(d_segmentOffsets);
            cudaFree(d_maxPerSegment);
            cudaFree(d_state);
            cudaFree(d_goesLeft);
            cudaFree(d_goesRight);
            cudaFree(d_statePrefixSum);
            cudaFree(d_newLabels);
            cudaFree(d_keepFlags);
            cudaFree(d_scatterIdx);
            cudaFree(d_labelCounts);
            cudaFree(d_labelOffsets);
            cudaFree(d_labelCounters);
            break;
        }
        
        float *d_px_new, *d_py_new;
        int *d_labels_new;
        cudaMalloc(&d_px_new, newN * sizeof(float));
        cudaMalloc(&d_py_new, newN * sizeof(float));
        cudaMalloc(&d_labels_new, newN * sizeof(int));
        
        compactPointsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_newLabels,
            d_px_new, d_py_new, d_labels_new,
            d_keepFlags, d_scatterIdx, currentN);
        cudaDeviceSynchronize();
        
        // Swap arrays
        cudaFree(d_px);
        cudaFree(d_py);
        cudaFree(d_labels);
        d_px = d_px_new;
        d_py = d_py_new;
        d_labels = d_labels_new;
        ansSize = newAnsSize;
        
        // Update counts
        currentN = newN;
        numLabels = newNumLabels;
        
        // Reallocate distances array if needed
        cudaFree(d_distances);
        cudaMalloc(&d_distances, currentN * sizeof(float));
        
        // Cleanup iteration-specific allocations
        cudaFree(d_segmentOffsets);
        cudaFree(d_maxPerSegment);
        cudaFree(d_state);
        cudaFree(d_goesLeft);
        cudaFree(d_goesRight);
        cudaFree(d_statePrefixSum);
        cudaFree(d_newLabels);
        cudaFree(d_keepFlags);
        cudaFree(d_scatterIdx);
        cudaFree(d_labelCounts);
        cudaFree(d_labelOffsets);
        cudaFree(d_labelCounters);
    }

    // Cleanup
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_ansX);
    cudaFree(d_ansY);

    // Return hull points (excluding endpoints which are added by caller)
    for (size_t i = 1; i < numLabels; i++) {
        hullPoints.push_back({h_ansX[i], h_ansY[i]});
    }
}


// ============================================================================
// Main entry point: runs QuickHull on upper and lower hulls separately
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

    // Copy points to device for CUB reduction
    float *d_px, *d_py;
    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);

    // Use GPU-accelerated min/max finding
    MinMaxPoint h_min, h_max;
    findMinMaxX_CUB(d_px, d_py, n, h_min, h_max);

    cudaFree(d_px);
    cudaFree(d_py);

    Point minPt = {h_min.x, h_min.y};
    Point maxPt = {h_max.x, h_max.y};

    // printf("Min Point: (%.3f, %.3f), Max Point: (%.3f, %.3f)\n", minPt.x, minPt.y, maxPt.x, maxPt.y);

    // Partition points into upper (above MIN->MAX line) and lower (below)
    std::vector<float> upperX, upperY, lowerX, lowerY;
    upperX.reserve(n);
    upperY.reserve(n);
    lowerX.reserve(n);
    lowerY.reserve(n);

    for (int i = 0; i < n; i++) {
        // Cross product to determine which side of MIN->MAX line
        float d = (maxPt.x - minPt.x) * (h_py[i] - minPt.y) - 
                  (maxPt.y - minPt.y) * (h_px[i] - minPt.x);
        if (d > 0) {
            // Above the line (upper hull)
            upperX.push_back(h_px[i]);
            upperY.push_back(h_py[i]);
        } else if (d < 0) {
            // Below the line (lower hull)
            lowerX.push_back(h_px[i]);
            lowerY.push_back(h_py[i]);
        }
        // d == 0: point is on the line, skip (collinear with endpoints)
    }

    // Find upper hull (points above MIN->MAX, going from MIN to MAX)
    std::vector<Point> upperHull;
    if (!upperX.empty()) {
        gpuQuickHullOneSide(upperX.data(), upperY.data(), upperX.size(),
                            minPt.x, minPt.y, maxPt.x, maxPt.y, upperHull);
    }

    // Find lower hull (points below MIN->MAX, going from MAX to MIN)
    std::vector<Point> lowerHull;
    if (!lowerX.empty()) {
        gpuQuickHullOneSide(lowerX.data(), lowerY.data(), lowerX.size(),
                            maxPt.x, maxPt.y, minPt.x, minPt.y, lowerHull);
    }

    // Combine: MIN -> upper hull -> MAX -> lower hull -> back to MIN
    std::vector<Point> hull;
    hull.push_back(minPt);
    for (auto &p : upperHull) {
        hull.push_back(p);
    }
    hull.push_back(maxPt);
    for (auto &p : lowerHull) {
        hull.push_back(p);
    }

    // Output
    *M = hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
}
