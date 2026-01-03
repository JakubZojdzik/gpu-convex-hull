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
    
    DistIdxPair identity{-FLT_MAX, -1};
    
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

// Structure for segmented max reduction - stores distance and original index
struct DistIdx {
    float dist;
    int idx;
};

// Reduction operator for finding max distance with index
struct MaxDistOp {
    __host__ __device__
    DistIdx operator()(const DistIdx &a, const DistIdx &b) const {
        // Only consider positive distances
        if (a.dist <= 0 && b.dist <= 0) return DistIdx{-FLT_MAX, -1};
        if (a.dist <= 0) return b;
        if (b.dist <= 0) return a;
        return (a.dist >= b.dist) ? a : b;
    }
};

// Kernel to build DistIdx array from distances
__global__ void buildDistIdxKernel(float *distances, DistIdx *distIdx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    distIdx[idx].dist = distances[idx];
    distIdx[idx].idx = idx;
}

// Kernel to detect segment heads (where label changes)
// Since points are sorted by label, segment head is where labels[i] != labels[i-1]
__global__ void detectSegmentHeadsKernel(int *labels, int *segmentHeads, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (idx == 0) {
        segmentHeads[0] = 1;  // First element is always a head
    } else {
        segmentHeads[idx] = (labels[idx] != labels[idx - 1]) ? 1 : 0;
    }
}

// Use CUB DeviceSegmentedReduce to find max per segment
void findMaxPerSegment(DistIdx *d_distIdx, int *d_labels, int n, int numLabels,
                       float *h_maxDist, int *h_maxIdx) {
    // First, we need to find segment boundaries
    // Since points are sorted by label, we can compute offsets
    
    // Count points per label by scanning
    int *d_segmentHeads;
    cudaMalloc(&d_segmentHeads, n * sizeof(int));
    
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    detectSegmentHeadsKernel<<<numBlocks, BLOCK_SIZE>>>(d_labels, d_segmentHeads, n);
    cudaDeviceSynchronize();
    
    // Compute segment offsets using exclusive scan
    int *d_segmentOffsets;
    cudaMalloc(&d_segmentOffsets, (n + 1) * sizeof(int));
    
    // Copy segment heads and do inclusive scan to get segment IDs
    int *d_segmentIds;
    cudaMalloc(&d_segmentIds, n * sizeof(int));
    
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_segmentHeads, d_segmentIds, n);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_segmentHeads, d_segmentIds, n);
    cudaFree(d_temp);
    
    // Now we need to find where each segment starts
    // Copy labels to host to build offset array
    std::vector<int> h_labels(n);
    cudaMemcpy(h_labels.data(), d_labels, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Build offsets array for CUB segmented reduce
    std::vector<int> h_offsets(numLabels + 1);
    h_offsets[0] = 0;
    int currentLabel = 0;
    for (int i = 0; i < n; i++) {
        while (currentLabel < h_labels[i]) {
            h_offsets[++currentLabel] = i;
        }
    }
    while (currentLabel < numLabels) {
        h_offsets[++currentLabel] = n;
    }
    h_offsets[numLabels] = n;
    
    // Copy offsets to device
    int *d_offsets;
    cudaMalloc(&d_offsets, (numLabels + 1) * sizeof(int));
    cudaMemcpy(d_offsets, h_offsets.data(), (numLabels + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    // Output array for per-segment max
    DistIdx *d_segmentMax;
    cudaMalloc(&d_segmentMax, numLabels * sizeof(DistIdx));
    
    // Run segmented reduce
    d_temp = nullptr;
    temp_bytes = 0;
    DistIdx identity{-FLT_MAX, -1};
    
    cub::DeviceSegmentedReduce::Reduce(
        d_temp, temp_bytes,
        d_distIdx, d_segmentMax,
        numLabels, d_offsets, d_offsets + 1,
        MaxDistOp(), identity);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSegmentedReduce::Reduce(
        d_temp, temp_bytes,
        d_distIdx, d_segmentMax,
        numLabels, d_offsets, d_offsets + 1,
        MaxDistOp(), identity);
    cudaFree(d_temp);
    
    // Copy results back
    std::vector<DistIdx> h_segmentMax(numLabels);
    cudaMemcpy(h_segmentMax.data(), d_segmentMax, numLabels * sizeof(DistIdx), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numLabels; i++) {
        h_maxDist[i] = h_segmentMax[i].dist;
        h_maxIdx[i] = h_segmentMax[i].idx;
    }
    
    // Cleanup
    cudaFree(d_segmentHeads);
    cudaFree(d_segmentOffsets);
    cudaFree(d_segmentIds);
    cudaFree(d_offsets);
    cudaFree(d_segmentMax);
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

// Kernel to mark points going to left or right sub-partition
// goesLeft[i] = 1 if point i goes to left sub-partition (label 2*oldLabel)
// goesRight[i] = 1 if point i goes to right sub-partition (label 2*oldLabel+1)
// eliminated[i] = 1 if point is eliminated (label -1)
__global__ void classifyForCompactionKernel(int *newLabels, 
                                             int *goesLeft, int *goesRight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = newLabels[idx];
    if (label < 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
    } else if (label % 2 == 0) {
        goesLeft[idx] = 1;
        goesRight[idx] = 0;
    } else {
        goesLeft[idx] = 0;
        goesRight[idx] = 1;
    }
}

// Kernel to compact and sort points by label using prefix scan results
// Points are written in order: all left partition points first, then all right partition points
// Within each group, they maintain their relative order (stable)
__global__ void compactSortedByLabelKernel(float *pxIn, float *pyIn, int *labelsIn,
                                            int *goesLeft, int *goesRight,
                                            int *leftScan, int *rightScan,
                                            int leftTotal,
                                            float *pxOut, float *pyOut, int *labelsOut,
                                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = labelsIn[idx];
    if (label < 0) return;  // Eliminated point
    
    int outIdx;
    if (goesLeft[idx]) {
        outIdx = leftScan[idx];
    } else {
        outIdx = leftTotal + rightScan[idx];
    }
    
    pxOut[outIdx] = pxIn[idx];
    pyOut[outIdx] = pyIn[idx];
    labelsOut[outIdx] = label;
}

// After compaction, we need to renumber labels to be contiguous (0, 1, 2, ...)
// and sort points within each old label group so left comes before right
// This kernel computes the final contiguous label
__global__ void renumberLabelsKernel(int *labels, int *labelMapping, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int oldLabel = labels[idx];
    labels[idx] = labelMapping[oldLabel];
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
    int maxAnsSize = n + 2;  // Upper bound on ANS size
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

        int numBlocks = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Compute distances for ALL points at once using labels
        size_t sharedMemSize = 2 * (BLOCK_SIZE + 2) * sizeof(float);
        computeDistancesKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_labels,
            d_ansX, d_ansY, (int)ans.size(),
            d_distances, currentN);
            
        cudaDeviceSynchronize();

        // Find max distance point for each partition using CUB segmented reduce
        // This follows the paper's methodology: since points are sorted by label,
        // we can use segmented operations efficiently
        int *d_segmentOffsets;
        DistIdxPair *d_maxPerSegment;
        cudaMalloc(&d_segmentOffsets, (numLabels + 1) * sizeof(int));
        cudaMalloc(&d_maxPerSegment, numLabels * sizeof(DistIdxPair));
        
        segmentedMaxDistReduce(d_distances, d_labels, d_segmentOffsets, 
                               d_maxPerSegment, currentN, numLabels);
        
        // Copy results back to host
        std::vector<DistIdxPair> h_maxPairs(numLabels);
        cudaMemcpy(h_maxPairs.data(), d_maxPerSegment, numLabels * sizeof(DistIdxPair), cudaMemcpyDeviceToHost);
        
        // Extract max distances and indices
        std::vector<float> h_maxDist(numLabels);
        std::vector<int> h_maxIdx(numLabels);
        for (int i = 0; i < numLabels; i++) {
            h_maxDist[i] = h_maxPairs[i].dist;
            h_maxIdx[i] = h_maxPairs[i].idx;
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
        
        int newLabel = 0;
        for (int i = 0; i < numLabels; i++) {
            newAns.push_back(ans[i]);
            labelMapping[i] = newLabel;
            
            if (h_maxIdx[i] >= 0 && h_maxDist[i] > 0) {
                // Get max point coordinates
                float maxPx, maxPy;
                cudaMemcpy(&maxPx, d_px + h_maxIdx[i], sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(&maxPy, d_py + h_maxIdx[i], sizeof(float), cudaMemcpyDeviceToHost);
                newAns.push_back({maxPx, maxPy});
                newLabel += 2;  // Two new partitions
            } else {
                newLabel += 1;  // Partition stays but gets renumbered
            }
        }
        newAns.push_back(ans.back());
        
        // Update labels on device
        int *d_newLabels;
        cudaMalloc(&d_newLabels, currentN * sizeof(int));
        
        updateLabelsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_labels,
            d_ansX, d_ansY, d_maxIdxPerLabel,
            d_newLabels, numLabels, currentN);
        cudaDeviceSynchronize();
        
        cudaFree(d_segmentOffsets);
        cudaFree(d_maxPerSegment);
        cudaFree(d_maxIdxPerLabel);
        
        // =====================================================================
        // Compact and sort points by label using prefix scans
        // This maintains the invariant that points are sorted by label
        // =====================================================================
        
        // Allocate arrays for compaction
        int *d_goesLeft, *d_goesRight;
        int *d_leftScan, *d_rightScan;
        float *d_pxNew, *d_pyNew;
        int *d_labelsNew;
        
        cudaMalloc(&d_goesLeft, currentN * sizeof(int));
        cudaMalloc(&d_goesRight, currentN * sizeof(int));
        cudaMalloc(&d_leftScan, currentN * sizeof(int));
        cudaMalloc(&d_rightScan, currentN * sizeof(int));
        cudaMalloc(&d_pxNew, currentN * sizeof(float));
        cudaMalloc(&d_pyNew, currentN * sizeof(float));
        cudaMalloc(&d_labelsNew, currentN * sizeof(int));
        
        // Classify points: left (even label), right (odd label), or eliminated (-1)
        classifyForCompactionKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_newLabels, d_goesLeft, d_goesRight, currentN);
        cudaDeviceSynchronize();
        
        // Prefix sums for compaction
        cubExclusiveScanInt(d_goesLeft, d_leftScan, currentN);
        cubExclusiveScanInt(d_goesRight, d_rightScan, currentN);
        
        // Get total counts
        int leftTotal, rightTotal;
        int lastLeft, lastRight;
        cudaMemcpy(&leftTotal, d_leftScan + currentN - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastLeft, d_goesLeft + currentN - 1, sizeof(int), cudaMemcpyDeviceToHost);
        leftTotal += lastLeft;
        cudaMemcpy(&rightTotal, d_rightScan + currentN - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastRight, d_goesRight + currentN - 1, sizeof(int), cudaMemcpyDeviceToHost);
        rightTotal += lastRight;
        
        int newN = leftTotal + rightTotal;
        
        if (newN == 0) {
            cudaFree(d_newLabels);
            cudaFree(d_goesLeft);
            cudaFree(d_goesRight);
            cudaFree(d_leftScan);
            cudaFree(d_rightScan);
            cudaFree(d_pxNew);
            cudaFree(d_pyNew);
            cudaFree(d_labelsNew);
            ans = newAns;
            break;
        }
        
        // Compact points, putting left partition points first, then right
        // This maintains sorted order by label since:
        // - Points with even labels (left partitions) come first
        // - Points with odd labels (right partitions) come after
        // - Within each group, relative order is preserved (stable)
        compactSortedByLabelKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_newLabels,
            d_goesLeft, d_goesRight,
            d_leftScan, d_rightScan, leftTotal,
            d_pxNew, d_pyNew, d_labelsNew,
            currentN);
        cudaDeviceSynchronize();
        
        // Now we need to renumber labels to be contiguous (0, 1, 2, ...)
        // Build label mapping: sparse label -> contiguous label
        // For the new ANS, partition i corresponds to label i
        // labelMapping maps the sparse label (2*oldLabel or 2*oldLabel+1) to new contiguous label
        int maxSparseLabel = 2 * numLabels;  // Upper bound on sparse labels
        std::vector<int> h_labelMapping(maxSparseLabel, -1);
        int contiguousLabel = 0;
        for (int i = 0; i < numLabels; i++) {
            if (h_maxIdx[i] >= 0 && h_maxDist[i] > 0) {
                // This partition split into two
                h_labelMapping[2 * i] = contiguousLabel++;      // Left sub-partition
                h_labelMapping[2 * i + 1] = contiguousLabel++;  // Right sub-partition
            }
            // If partition didn't split, no points remain in it
        }
        
        // Copy label mapping to device and renumber
        int *d_labelMapping;
        cudaMalloc(&d_labelMapping, maxSparseLabel * sizeof(int));
        cudaMemcpy(d_labelMapping, h_labelMapping.data(), maxSparseLabel * sizeof(int), cudaMemcpyHostToDevice);
        
        int newNumBlocks = (newN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        renumberLabelsKernel<<<newNumBlocks, BLOCK_SIZE>>>(d_labelsNew, d_labelMapping, newN);
        cudaDeviceSynchronize();
        
        cudaFree(d_labelMapping);
        
        // Swap buffers
        cudaMemcpy(d_px, d_pxNew, newN * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_py, d_pyNew, newN * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_labels, d_labelsNew, newN * sizeof(int), cudaMemcpyDeviceToDevice);
        
        // Cleanup temp arrays
        cudaFree(d_newLabels);
        cudaFree(d_goesLeft);
        cudaFree(d_goesRight);
        cudaFree(d_leftScan);
        cudaFree(d_rightScan);
        cudaFree(d_pxNew);
        cudaFree(d_pyNew);
        cudaFree(d_labelsNew);
        
        currentN = newN;
        ans = newAns;
        numLabels = contiguousLabel;
    }

    // Cleanup
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_ansX);
    cudaFree(d_ansY);

    // Return hull points (excluding endpoints which are added by caller)
    for (size_t i = 1; i < ans.size() - 1; i++) {
        hullPoints.push_back(ans[i]);
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
