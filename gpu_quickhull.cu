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

// ============================================================================
// Simple QuickHull for ONE side of the hull (upper or lower)
// All points have label=0, single partition that recursively splits
// ============================================================================

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

// ============================================================================
// Structs and operators for segmented max distance reduction (per paper methodology)
// ============================================================================

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
    
    // First thread sets first segment start
    if (i == 0) {
        offsets[0] = 0;
        offsets[numSegments] = n;  // Sentinel
    }
    
    // Each thread checks if there's a segment boundary at position i+1
    if (i < n - 1) {
        if (labels[i] != labels[i + 1]) {
            // Boundary between segment labels[i] and labels[i+1]
            offsets[labels[i + 1]] = i + 1;
        }
    }
}


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
    
    // Build DistIdxPair array
    DistIdxPair *d_pairs;
    cudaMalloc(&d_pairs, n * sizeof(DistIdxPair));
    buildDistIdxArray<<<numBlocks, BLOCK_SIZE>>>(d_distances, d_pairs, n);
    cudaDeviceSynchronize();
    
    // Use CUB segmented reduce to find max per segment
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    
    DistIdxPair identity{0, -1};
    
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
            printf("Error: ans index out of range in computeDistancesKernel\n");
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

// Kernel to update labels after finding max points
// For each point, if its partition got a max point, update label based on which side
// Points with label -1 will be eliminated in compaction
__global__ void updateLabelsKernel(float *px, float *py, int *labels,
                                    float *ansX, float *ansY, int *maxIdxPerLabel,
                                    int *newLabels, int numLabels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = labels[idx];
    int maxIdx = maxIdxPerLabel[label];
    
    if (maxIdx < 0) {
        // No max point found for this partition, point is eliminated
        newLabels[idx] = -1;
        return;
    }
    
    // Get L, M, R points
    float lx = ansX[label];
    float ly = ansY[label];
    float rx = ansX[label + 1];
    float ry = ansY[label + 1];
    float mx = px[maxIdx];
    float my = py[maxIdx];
    
    // Skip the max point itself
    if (idx == maxIdx) {
        newLabels[idx] = -1;
        return;
    }
    
    float curX = px[idx];
    float curY = py[idx];
    
    // Distance from L->M line
    float distLM = (mx - lx) * (curY - ly) - (my - ly) * (curX - lx);
    // Distance from M->R line  
    float distMR = (rx - mx) * (curY - my) - (ry - my) * (curX - mx);
    
    // New label: 2*label for left partition (L->M), 2*label+1 for right partition (M->R)
    if (distLM > 0) {
        newLabels[idx] = 2 * label;
    } else if (distMR > 0) {
        newLabels[idx] = 2 * label + 1;
    } else {
        newLabels[idx] = -1;  // Point is inside the new triangle, eliminate
    }
}

// Kernel to create survive flags (1 if label >= 0, else 0)
__global__ void createSurviveFlagsKernel(int *labels, int *survives, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    survives[idx] = (labels[idx] >= 0) ? 1 : 0;
}

// Simple compaction kernel - removes eliminated points, keeps relative order
__global__ void compactKernel(float *pxIn, float *pyIn, int *labelsIn,
                              int *scanResult,
                              float *pxOut, float *pyOut, int *labelsOut,
                              int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (labelsIn[idx] >= 0) {
        int outIdx = scanResult[idx];
        pxOut[outIdx] = pxIn[idx];
        pyOut[outIdx] = pyIn[idx];
        labelsOut[outIdx] = labelsIn[idx];
    }
}

// Renumber labels using a mapping table
__global__ void renumberLabelsKernel(int *labels, int *labelMapping, int maxLabel, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int oldLabel = labels[idx];
    if (oldLabel >= 0 && oldLabel < maxLabel) {
        labels[idx] = labelMapping[oldLabel];
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
    printf("\n========== gpuQuickHullOneSide START ==========\n");
    printf("n=%d points, left=(%.3f, %.3f), right=(%.3f, %.3f)\n", n, leftX, leftY, rightX, rightY);
    
    if (n == 0) {
        printf("No points, returning empty hull.\n");
        return;
    }

    // Allocate device memory
    float *d_px, *d_py;
    float *d_distances;
    int *d_labels;
    float *d_ansX, *d_ansY;

    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMalloc(&d_distances, n * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));

    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Initialize all labels to 0 (single partition)
    cudaMemset(d_labels, 0, n * sizeof(int));

    // ANS stores the hull points in order (on host, copied to device as needed)
    std::vector<Point> ans;
    ans.push_back({leftX, leftY});
    ans.push_back({rightX, rightY});
    
    int currentN = n;
    int numLabels = 1;  // Start with 1 partition (label 0)
    
    // Allocate ANS arrays on device (will grow as needed)
    int maxAnsSize = n + 1;  // Upper bound on ANS size
    cudaMalloc(&d_ansX, maxAnsSize * sizeof(float));
    cudaMalloc(&d_ansY, maxAnsSize * sizeof(float));

    while (true) {
        // Copy current ANS to device
        std::vector<float> ansX(ans.size()), ansY(ans.size());
        for (size_t i = 0; i < ans.size(); i++) {
            ansX[i] = ans[i].x;
            ansY[i] = ans[i].y;
        }
        cudaMemcpy(d_ansX, ansX.data(), ans.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ansY, ansY.data(), ans.size() * sizeof(float), cudaMemcpyHostToDevice);

        printf("\n=== ITERATION START: currentN=%d, numLabels=%d, ans.size=%zu ===\n", currentN, numLabels, ans.size());
        printf("ANS points:\n");
        for (size_t i = 0; i < ans.size(); i++) {
            printf("  ANS[%zu] = (%.3f, %.3f)\n", i, ans[i].x, ans[i].y);
        }

        // Debug: print current points and labels
        std::vector<float> dbg_px(currentN), dbg_py(currentN);
        std::vector<int> dbg_labels(currentN);
        cudaMemcpy(dbg_px.data(), d_px, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg_py.data(), d_py, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg_labels.data(), d_labels, currentN * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Current points (before distance computation):\n");
        for (int i = 0; i < currentN && i < 50; i++) {
            printf("  Point[%d] = (%.3f, %.3f), label=%d\n", i, dbg_px[i], dbg_py[i], dbg_labels[i]);
        }
        if (currentN > 50) printf("  ... (%d more points)\n", currentN - 50);

        int numBlocks = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Compute distances for ALL points at once using labels
        size_t sharedMemSize = 2 * (BLOCK_SIZE + 2) * sizeof(float);
        computeDistancesKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_labels,
            d_ansX, d_ansY, (int)ans.size(),
            d_distances, currentN);
            
        cudaDeviceSynchronize();

        // Debug: print distances
        std::vector<float> dbg_distances(currentN);
        cudaMemcpy(dbg_distances.data(), d_distances, currentN * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Distances after computation:\n");
        for (int i = 0; i < currentN && i < 50; i++) {
            printf("  Distance[%d] = %.3f (label=%d)\n", i, dbg_distances[i], dbg_labels[i]);
        }
        if (currentN > 50) printf("  ... (%d more distances)\n", currentN - 50);

        // Find max distance point for each partition using CUB segmented reduce
        // This follows the paper's methodology: since points are sorted by label,
        // we can use segmented operations efficiently
        int *d_segmentOffsets;
        DistIdxPair *d_maxPerSegment;
        cudaMalloc(&d_segmentOffsets, (numLabels + 1) * sizeof(int));
        cudaMalloc(&d_maxPerSegment, numLabels * sizeof(DistIdxPair));
        
        segmentedMaxDistReduce(d_distances, d_labels, d_segmentOffsets, 
                               d_maxPerSegment, currentN, numLabels);
        
        // Debug: print segment offsets
        std::vector<int> dbg_offsets(numLabels + 1);
        cudaMemcpy(dbg_offsets.data(), d_segmentOffsets, (numLabels + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        printf("Segment offsets:\n");
        for (int i = 0; i <= numLabels; i++) {
            printf("  offset[%d] = %d\n", i, dbg_offsets[i]);
        }

        // Copy results back to host
        std::vector<DistIdxPair> h_maxPairs(numLabels);
        cudaMemcpy(h_maxPairs.data(), d_maxPerSegment, numLabels * sizeof(DistIdxPair), cudaMemcpyDeviceToHost);
        
        // Extract max distances and indices
        std::vector<float> h_maxDist(numLabels);
        std::vector<int> h_maxIdx(numLabels);
        for (int i = 0; i < numLabels; i++) {
            h_maxDist[i] = h_maxPairs[i].dist;
            h_maxIdx[i] = h_maxPairs[i].idx;
            printf("Max for label %d: dist=%.3f, idx=%d\n", i, h_maxDist[i], h_maxIdx[i]);
        }
        
        // Check if any partition found a max point
        bool anyChanged = false;
        for (int i = 0; i < numLabels; i++) {
            if (h_maxIdx[i] >= 0 && h_maxDist[i] > 0) {
                anyChanged = true;
                break;
            }
        }
        
        if (!anyChanged) {
            printf("No partition changed, terminating loop.\n");
            cudaFree(d_segmentOffsets);
            cudaFree(d_maxPerSegment);
            break;
        }
        
        // Build new ANS by inserting max points
        std::vector<Point> newAns;
        std::vector<int> labelMapping(numLabels);  // Maps old label to position in new ANS
        
        // Need maxIdx per label for updateLabelsKernel - copy to device
        int *d_maxIdxPerLabel;
        cudaMalloc(&d_maxIdxPerLabel, numLabels * sizeof(int));
        cudaMemcpy(d_maxIdxPerLabel, h_maxIdx.data(), numLabels * sizeof(int), cudaMemcpyHostToDevice);
        
        printf("Building newAns:\n");
        int newLabel = 0;
        for (int i = 0; i < numLabels; i++) {
            newAns.push_back(ans[i]);
            printf("  newAns[%zu] = ans[%d] = (%.3f, %.3f)\n", newAns.size()-1, i, ans[i].x, ans[i].y);
            labelMapping[i] = newLabel;
            
            if (h_maxIdx[i] >= 0 && h_maxDist[i] > 0) {
                // Get max point coordinates
                float maxPx, maxPy;
                cudaMemcpy(&maxPx, d_px + h_maxIdx[i], sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&maxPy, d_py + h_maxIdx[i], sizeof(float), cudaMemcpyDeviceToHost);
                newAns.push_back({maxPx, maxPy});
                printf("  newAns[%zu] = MAX POINT from partition %d: (%.3f, %.3f)\n", newAns.size()-1, i, maxPx, maxPy);
                printf("    -> sparse labels for this partition: %d (left), %d (right)\n", 2*i, 2*i+1);
                newLabel += 2;  // Two new partitions
            } else {
                printf("  Partition %d did NOT split (no valid max)\n", i);
                newLabel += 1;  // Partition stays but gets renumbered
            }
        }
        newAns.push_back(ans.back());
        printf("  newAns[%zu] = ans.back() = (%.3f, %.3f)\n", newAns.size()-1, ans.back().x, ans.back().y);
        
        // Update labels on device
        int *d_newLabels;
        cudaMalloc(&d_newLabels, currentN * sizeof(int));
        
        updateLabelsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels,
            d_ansX, d_ansY, d_maxIdxPerLabel,
            d_newLabels, numLabels, currentN);
        cudaDeviceSynchronize();
        
        // Debug: print new labels after updateLabelsKernel
        std::vector<int> dbg_newLabels(currentN);
        cudaMemcpy(dbg_newLabels.data(), d_newLabels, currentN * sizeof(int), cudaMemcpyDeviceToHost);
        printf("After updateLabelsKernel (sparse labels, -1 = eliminated):\n");
        for (int i = 0; i < currentN && i < 50; i++) {
            printf("  Point[%d] (%.3f, %.3f): oldLabel=%d -> newLabel=%d\n", 
                   i, dbg_px[i], dbg_py[i], dbg_labels[i], dbg_newLabels[i]);
        }
        if (currentN > 50) printf("  ... (%d more points)\n", currentN - 50);
        
        cudaFree(d_segmentOffsets);
        cudaFree(d_maxPerSegment);
        cudaFree(d_maxIdxPerLabel);
        
        // =====================================================================
        // Compact points: remove eliminated (label == -1)
        // Then renumber labels to be contiguous based on surviving points
        // =====================================================================
        
        // Allocate arrays for compaction
        int *d_survives;
        int *d_scanResult;
        float *d_pxNew, *d_pyNew;
        int *d_labelsNew;
        
        cudaMalloc(&d_survives, currentN * sizeof(int));
        cudaMalloc(&d_scanResult, currentN * sizeof(int));
        cudaMalloc(&d_pxNew, currentN * sizeof(float));
        cudaMalloc(&d_pyNew, currentN * sizeof(float));
        cudaMalloc(&d_labelsNew, currentN * sizeof(int));
        
        // Create survive flags (1 if label >= 0)
        createSurviveFlagsKernel<<<numBlocks, BLOCK_SIZE>>>(d_newLabels, d_survives, currentN);
        cudaDeviceSynchronize();
        
        // Prefix sum for compaction
        cubExclusiveScanInt(d_survives, d_scanResult, currentN);
        
        // Get total count of surviving points
        int lastSurvive, lastScan;
        cudaMemcpy(&lastSurvive, d_survives + currentN - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastScan, d_scanResult + currentN - 1, sizeof(int), cudaMemcpyDeviceToHost);
        int newN = lastScan + lastSurvive;
        
        printf("Compaction: %d points -> %d surviving points\n", currentN, newN);
        
        if (newN == 0) {
            printf("All points eliminated, terminating loop.\n");
            cudaFree(d_newLabels);
            cudaFree(d_survives);
            cudaFree(d_scanResult);
            cudaFree(d_pxNew);
            cudaFree(d_pyNew);
            cudaFree(d_labelsNew);
            ans = newAns;
            break;
        }
        
        // Compact points (remove eliminated ones, keep sparse labels for now)
        compactKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_newLabels,
            d_scanResult,
            d_pxNew, d_pyNew, d_labelsNew,
            currentN);
        cudaDeviceSynchronize();
        
        cudaFree(d_survives);
        cudaFree(d_scanResult);
        
        // Copy compacted labels to host to build proper label mapping
        std::vector<int> h_compactedLabels(newN);
        cudaMemcpy(h_compactedLabels.data(), d_labelsNew, newN * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Debug: print compacted points with sparse labels
        std::vector<float> dbg_pxNew(newN), dbg_pyNew(newN);
        cudaMemcpy(dbg_pxNew.data(), d_pxNew, newN * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(dbg_pyNew.data(), d_pyNew, newN * sizeof(float), cudaMemcpyDeviceToHost);
        printf("After compaction (sparse labels):\n");
        for (int i = 0; i < newN && i < 50; i++) {
            printf("  Point[%d] = (%.3f, %.3f), sparseLabel=%d\n", i, dbg_pxNew[i], dbg_pyNew[i], h_compactedLabels[i]);
        }
        if (newN > 50) printf("  ... (%d more points)\n", newN - 50);
        
        // Find which sparse labels actually have points
        int maxSparseLabel = 2 * numLabels;
        std::vector<bool> labelExists(maxSparseLabel, false);
        for (int i = 0; i < newN; i++) {
            if (h_compactedLabels[i] >= 0 && h_compactedLabels[i] < maxSparseLabel) {
                labelExists[h_compactedLabels[i]] = true;
            }
        }
        
        printf("Sparse labels that exist: ");
        for (int i = 0; i < maxSparseLabel; i++) {
            if (labelExists[i]) printf("%d ", i);
        }
        printf("\n");
        
        // Build mapping from sparse labels to contiguous labels
        // AND build the new ANS to match
        std::vector<int> h_labelMapping(maxSparseLabel, -1);
        std::vector<Point> finalNewAns;
        finalNewAns.push_back(newAns[0]);  // First point (leftPt)
        printf("Building finalNewAns and label mapping:\n");
        printf("  finalNewAns[0] = (%.3f, %.3f) [leftPt]\n", newAns[0].x, newAns[0].y);
        
        int contiguousLabel = 0;
        for (int sparseLabel = 0; sparseLabel < maxSparseLabel; sparseLabel++) {
            if (labelExists[sparseLabel]) {
                h_labelMapping[sparseLabel] = contiguousLabel;
                printf("  sparseLabel %d -> contiguousLabel %d\n", sparseLabel, contiguousLabel);
                // Find the corresponding ANS point for this sparse label
                // Sparse label i corresponds to ANS[i] -> ANS[i+1] edge
                // We need to add ANS[sparseLabel + 1] to finalNewAns
                if (sparseLabel + 1 < (int)newAns.size()) {
                    finalNewAns.push_back(newAns[sparseLabel + 1]);
                    printf("  finalNewAns[%zu] = newAns[%d] = (%.3f, %.3f)\n", 
                           finalNewAns.size()-1, sparseLabel + 1, newAns[sparseLabel + 1].x, newAns[sparseLabel + 1].y);
                }
                contiguousLabel++;
            }
        }
        
        printf("Final ANS for next iteration (%zu points):\n", finalNewAns.size());
        for (size_t i = 0; i < finalNewAns.size(); i++) {
            printf("  finalNewAns[%zu] = (%.3f, %.3f)\n", i, finalNewAns[i].x, finalNewAns[i].y);
        }
        
        // Copy label mapping to device and renumber
        int *d_labelMapping;
        cudaMalloc(&d_labelMapping, maxSparseLabel * sizeof(int));
        cudaMemcpy(d_labelMapping, h_labelMapping.data(), maxSparseLabel * sizeof(int), cudaMemcpyHostToDevice);
        
        int newNumBlocks = (newN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        renumberLabelsKernel<<<newNumBlocks, BLOCK_SIZE>>>(d_labelsNew, d_labelMapping, maxSparseLabel, newN);
        cudaDeviceSynchronize();
        
        // Debug: print final renumbered labels
        std::vector<int> dbg_finalLabels(newN);
        cudaMemcpy(dbg_finalLabels.data(), d_labelsNew, newN * sizeof(int), cudaMemcpyDeviceToHost);
        printf("After renumbering (contiguous labels):\n");
        for (int i = 0; i < newN && i < 50; i++) {
            printf("  Point[%d] = (%.3f, %.3f), finalLabel=%d\n", i, dbg_pxNew[i], dbg_pyNew[i], dbg_finalLabels[i]);
        }
        if (newN > 50) printf("  ... (%d more points)\n", newN - 50);
        
        cudaFree(d_labelMapping);
        
        // Swap buffers
        cudaMemcpy(d_px, d_pxNew, newN * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_py, d_pyNew, newN * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_labels, d_labelsNew, newN * sizeof(int), cudaMemcpyDeviceToDevice);
        
        // Cleanup temp arrays
        cudaFree(d_newLabels);
        cudaFree(d_pxNew);
        cudaFree(d_pyNew);
        cudaFree(d_labelsNew);
        
        currentN = newN;
        ans = finalNewAns;
        numLabels = contiguousLabel;
        
        printf("=== ITERATION END: currentN=%d, numLabels=%d ===\n\n", currentN, numLabels);
    }

    printf("=== LOOP TERMINATED ===\n");
    printf("Final ANS (%zu points):\n", ans.size());
    for (size_t i = 0; i < ans.size(); i++) {
        printf("  ans[%zu] = (%.3f, %.3f)\n", i, ans[i].x, ans[i].y);
    }

    // Cleanup
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_ansX);
    cudaFree(d_ansY);

    // Return hull points (excluding endpoints which are added by caller)
    printf("Returning hull points (excluding endpoints):\n");
    for (size_t i = 1; i < ans.size() - 1; i++) {
        hullPoints.push_back(ans[i]);
        printf("  hullPoints[%zu] = (%.3f, %.3f)\n", hullPoints.size()-1, ans[i].x, ans[i].y);
    }
    printf("========== gpuQuickHullOneSide END ==========\n\n");
}


// ============================================================================
// Main entry point: runs QuickHull on upper and lower hulls separately
// ============================================================================
extern "C" void gpuQuickHull(float *h_px, float *h_py, int n,
                              float *result_x, float *result_y, int *M) {
    printf("\n############### gpuQuickHull START ###############\n");
    printf("Total input points: %d\n", n);
    
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
    
    printf("Min point: (%.3f, %.3f), Max point: (%.3f, %.3f)\n", minPt.x, minPt.y, maxPt.x, maxPt.y);

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
    
    printf("Partitioned: %zu upper points, %zu lower points\n", upperX.size(), lowerX.size());

    // Find upper hull (points above MIN->MAX, going from MIN to MAX)
    printf("\n--- Processing UPPER hull ---\n");
    std::vector<Point> upperHull;
    if (!upperX.empty()) {
        gpuQuickHullOneSide(upperX.data(), upperY.data(), upperX.size(),
                            minPt.x, minPt.y, maxPt.x, maxPt.y, upperHull);
    }
    printf("Upper hull has %zu points (excluding endpoints)\n", upperHull.size());

    // Find lower hull (points below MIN->MAX, going from MAX to MIN)
    printf("\n--- Processing LOWER hull ---\n");
    std::vector<Point> lowerHull;
    if (!lowerX.empty()) {
        gpuQuickHullOneSide(lowerX.data(), lowerY.data(), lowerX.size(),
                            maxPt.x, maxPt.y, minPt.x, minPt.y, lowerHull);
    }
    printf("Lower hull has %zu points (excluding endpoints)\n", lowerHull.size());

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
    
    printf("\n--- Final combined hull ---\n");
    for (size_t i = 0; i < hull.size(); i++) {
        printf("hull[%zu] = (%.3f, %.3f)\n", i, hull[i].x, hull[i].y);
    }

    // Output
    *M = hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
    
    printf("############### gpuQuickHull END ###############\n\n");
}
