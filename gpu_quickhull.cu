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

#define DEBUG_PRINT
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
struct DistIdxPair {
    float dist;
    int   idx;
};

struct MaxDistOp {
    __host__ __device__
    DistIdxPair operator()(const DistIdxPair &a, const DistIdxPair &b) const {
        return (a.dist >= b.dist) ? a : b;
    }
};

__global__ void buildDistIdxArray(const float *distances, DistIdxPair *pairs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pairs[i].dist = distances[i];
        pairs[i].idx = i;
    }
}

__global__ void fillOffsets(int *offsets, int numSegments) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < numSegments && i > 0 && offsets[i] == -1 && offsets[i+1] != -1) {
        for(int j = i; offsets[j] == -1 && j > 0; j--) {
            offsets[j] = offsets[i+1];
        }
    }
}

__global__ void findSegmentOffsetsKernel(const int *labels, int *offsets, int numSegments, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i == 0) {
        offsets[numSegments] = n;  // Sentinel
        offsets[labels[0]] = 0;
    }
    
    if (i < n - 1 && labels[i] != labels[i + 1])
        offsets[labels[i + 1]] = i + 1;
}


void segmentedMaxDistReduce(
    const float *d_distances,
    const int *d_labels,
    int *d_segmentOffsets,
    DistIdxPair *d_maxPerSegment,
    int n,
    int numSegments)
{
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    findSegmentOffsetsKernel<<<numBlocks, BLOCK_SIZE>>>(
        d_labels, d_segmentOffsets, numSegments, n);
    cudaDeviceSynchronize();

    int numBlocksSegments = (numSegments + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fillOffsets<<<numBlocksSegments, BLOCK_SIZE>>>(d_segmentOffsets, numSegments);

    DistIdxPair *d_pairs;
    cudaMalloc(&d_pairs, n * sizeof(DistIdxPair));
    buildDistIdxArray<<<numBlocks, BLOCK_SIZE>>>(d_distances, d_pairs, n);
    
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    
    DistIdxPair identity{0.0f, -1};
    cudaDeviceSynchronize();
    
    cub::DeviceSegmentedReduce::Reduce(
        d_temp, temp_bytes,
        d_pairs, d_maxPerSegment,
        numSegments,
        d_segmentOffsets, d_segmentOffsets + 1,
        MaxDistOp(), identity);
    
    cudaMalloc(&d_temp, temp_bytes);
    
    cub::DeviceSegmentedReduce::Reduce(
        d_temp, temp_bytes,
        d_pairs, d_maxPerSegment,
        numSegments,
        d_segmentOffsets, d_segmentOffsets + 1,
        MaxDistOp(), identity);
    
    cudaFree(d_temp);
    cudaFree(d_pairs);
}


__global__ void computeDistancesKernel(float *px, float *py, int *labels,
                                        float *ansX, float *ansY, int ansSize,
                                        float *distances, int n) {
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
    // indexes [minLabel, maxLabel+1] from ANS
    int ansRange = maxLabel - minLabel + 2;
    if (tid < ansRange) {
        if ((minLabel + tid) >= ansSize) {
            return;
        }
        sAnsX[tid] = ansX[minLabel + tid];
        sAnsY[tid] = ansY[minLabel + tid];
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

    float d = (rx - lx) * (curY - ly) - (ry - ly) * (curX - lx);
    distances[idx] = d;
}

static void cubExclusiveScanInt(int *d_input, int *d_output, int n) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaFree(d_temp_storage);
}

__global__ void determineSide(float *px, float *py, int *labels,
                                    float *ansX, float *ansY, int *state, DistIdxPair *maxIdxPerLabel,
                                    int *goesLeft, int *goesRight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int label = labels[idx];
    int maxIdx = maxIdxPerLabel[label].idx;
    
    if (maxIdx < 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        state[label] = 0;
        return;
    }

    if (maxIdx == idx) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        state[label] = 1;
        return;
    }

    float lx = ansX[label];
    float ly = ansY[label];
    float rx = ansX[label + 1];
    float ry = ansY[label + 1];
    float mx = px[maxIdx];
    float my = py[maxIdx];
    float curX = px[idx];
    float curY = py[idx];
    
    float distLM = (mx - lx) * (curY - ly) - (my - ly) * (curX - lx);
    float distMR = (rx - mx) * (curY - my) - (ry - my) * (curX - mx);
    
    if (distLM > 0) {
        goesLeft[idx] = 1;
        goesRight[idx] = 0;
    } else if (distMR > 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 1;
    }
}

__global__ void computeNewLabelsKernel(int *labels, int *statePrefixSum,
                                        int *goesLeft, int *goesRight,
                                        int *newLabels, int *keepFlags, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int keep = goesLeft[idx] | goesRight[idx];
    keepFlags[idx] = keep;
    
    if (keep) {
        int label = labels[idx];
        newLabels[idx] = label + statePrefixSum[label] + goesRight[idx];
    } else {
        newLabels[idx] = INT_MAX;
    }
}

__global__ void countPerLabelKernel(int *newLabels, int *keepFlags, 
                                     int *labelCounts, int n, int newNumLabels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (keepFlags[idx]) {
        int newLabel = newLabels[idx];
        if (newLabel >= 0 && newLabel < newNumLabels) {
            atomicAdd(&labelCounts[newLabel], 1);
        }
    }
}

__global__ void computeLocalScatterIdx(int *newLabels, int *keepFlags,
                                        int *labelOffsets, int *scatterIdx, 
                                        int *labelCounters, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    if (keepFlags[idx]) {
        int newLabel = newLabels[idx];
        int localIdx = atomicAdd(&labelCounters[newLabel], 1);
        scatterIdx[idx] = labelOffsets[newLabel] + localIdx;
    }
}

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

__global__ void buildNewAnsKernel(float *oldAnsX, float *oldAnsY,
                                   float *newAnsX, float *newAnsY,
                                   float *px, float *py,
                                   int *state, int *statePrefixSum,
                                   DistIdxPair *maxPerSegment,
                                   int numLabels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > numLabels) return;
    
    if (i < numLabels) {
        int newPos = i + statePrefixSum[i];
        newAnsX[newPos] = oldAnsX[i];
        newAnsY[newPos] = oldAnsY[i];
        
        // partition splits
        if (state[i] == 1 && maxPerSegment[i].idx >= 0) {
            int maxIdx = maxPerSegment[i].idx;
            newAnsX[newPos + 1] = px[maxIdx];
            newAnsY[newPos + 1] = py[maxIdx];
        }
    } else {
        // i == numLabels, final endpoint
        int totalInserted = statePrefixSum[numLabels - 1] + state[numLabels - 1];
        int newPos = numLabels + totalInserted;
        newAnsX[newPos] = oldAnsX[numLabels];
        newAnsY[newPos] = oldAnsY[numLabels];
    }
}


static void gpuQuickHullOneSide(float *h_px, float *h_py, int n,
                          float leftX, float leftY, float rightX, float rightY,
                          std::vector<Point> &hullPoints) {
    if (n == 0) return;

    float *d_px, *d_py;
    float *d_distances;
    int *d_labels;
    float *d_ansX, *d_ansY;
    int *d_state;
    int *d_goesLeft, *d_goesRight;
    float *d_newAnsX, *d_newAnsY;
    float *d_px_new, *d_py_new;
    int *d_labels_new;
    int maxAnsSize = n + 1;

    float *h_ansX, *h_ansY;
    h_ansX = (float*)malloc(maxAnsSize * sizeof(float));
    h_ansY = (float*)malloc(maxAnsSize * sizeof(float));
    h_ansX[0] = leftX;
    h_ansY[0] = leftY;
    h_ansX[1] = rightX;
    h_ansY[1] = rightY;

    int currentN = n;
    int numLabels = 1;
    int ansSize = 2;

    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMalloc(&d_distances, n * sizeof(float));
    cudaMalloc(&d_labels, n * sizeof(int));

    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_labels, 0, n * sizeof(int));

    cudaMalloc(&d_ansX, maxAnsSize * sizeof(float));
    cudaMalloc(&d_ansY, maxAnsSize * sizeof(float));
    cudaMemcpy(d_ansX, h_ansX, ansSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ansY, h_ansY, ansSize * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_state, maxAnsSize * sizeof(int));
    cudaMalloc(&d_goesLeft, n * sizeof(int));
    cudaMalloc(&d_goesRight, n * sizeof(int));

    cudaMalloc(&d_segmentOffsets, (maxAnsSize + 1) * sizeof(int));
    cudaMalloc(&d_maxPerSegment, maxAnsSize * sizeof(DistIdxPair));
    cudaMalloc(&d_statePrefixSum, maxAnsSize * sizeof(int));

    cudaMalloc(&d_newLabels, n * sizeof(int));
    cudaMalloc(&d_keepFlags, n * sizeof(int));
    cudaMalloc(&d_scatterIdx, n * sizeof(int));

    cudaMalloc(&d_labelCounts, maxAnsSize * sizeof(int));
    cudaMalloc(&d_labelOffsets, (maxAnsSize + 1) * sizeof(int));
    cudaMalloc(&d_labelCounters, maxAnsSize * sizeof(int));

    cudaMalloc(&d_newAnsX, maxAnsSize * sizeof(float));
    cudaMalloc(&d_newAnsY, maxAnsSize * sizeof(float));
    
    cudaMalloc(&d_px_new, n * sizeof(float));
    cudaMalloc(&d_py_new, n * sizeof(float));
    cudaMalloc(&d_labels_new, n * sizeof(int));

    free(h_ansX);
    free(h_ansY);

    while (true) {
        int numBlocks = (currentN + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        size_t sharedMemSize = 2 * (BLOCK_SIZE + 2) * sizeof(float);
        computeDistancesKernel<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
            d_px, d_py, d_labels,
            d_ansX, d_ansY, ansSize,
            d_distances, currentN);

        int *d_segmentOffsets;
        DistIdxPair *d_maxPerSegment;
        cudaMemset(d_segmentOffsets, -1, (numLabels + 1) * sizeof(int));
        
        segmentedMaxDistReduce(d_distances, d_labels, d_segmentOffsets, 
                               d_maxPerSegment, currentN, numLabels);
        cudaDeviceSynchronize();

        cudaMemset(d_state, 0, numLabels * sizeof(int));
        cudaMemset(d_goesLeft, 0, currentN * sizeof(int));
        cudaMemset(d_goesRight, 0, currentN * sizeof(int));
        determineSide<<<numBlocks, BLOCK_SIZE>>>(d_px, d_py, d_labels, d_ansX, d_ansY, d_state, d_maxPerSegment, d_goesLeft, d_goesRight, currentN);

        int *d_statePrefixSum;
        cubExclusiveScanInt(d_state, d_statePrefixSum, numLabels);

        int lastState, lastStatePrefixSum;
        cudaMemcpy(&lastState, d_state + numLabels - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lastStatePrefixSum, d_statePrefixSum + numLabels - 1, sizeof(int), cudaMemcpyDeviceToHost);
        int numNewHullPoints = lastState + lastStatePrefixSum;

        int newNumLabels = numLabels + numNewHullPoints;

        if (numNewHullPoints == 0) {
            break;
        }
        
        int *d_newLabels, *d_keepFlags, *d_scatterIdx;
        cudaMemset(d_keepFlags, 0, currentN * sizeof(int));
        
        computeNewLabelsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_labels, d_statePrefixSum,
            d_goesLeft, d_goesRight,
            d_newLabels, d_keepFlags, currentN);
        
        int *d_labelCounts, *d_labelOffsets, *d_labelCounters;
        cudaMemset(d_labelCounts, 0, newNumLabels * sizeof(int));
        cudaMemset(d_labelCounters, 0, newNumLabels * sizeof(int));
        
        countPerLabelKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_newLabels, d_keepFlags, d_labelCounts, currentN, newNumLabels);
        
        cubExclusiveScanInt(d_labelCounts, d_labelOffsets, newNumLabels + 1);

        int newN;
        cudaMemcpy(&newN, d_labelOffsets + newNumLabels, sizeof(int), cudaMemcpyDeviceToHost);
        int newAnsSize = numLabels + 1 + newNumLabels - numLabels;
        
        int ansBlocks = (numLabels + 1 + BLOCK_SIZE) / BLOCK_SIZE;
        buildNewAnsKernel<<<ansBlocks, BLOCK_SIZE>>>(
            d_ansX, d_ansY, d_newAnsX, d_newAnsY,
            d_px, d_py, d_state, d_statePrefixSum, d_maxPerSegment, numLabels);
        
        cudaMemcpy(d_ansX, d_newAnsX, newAnsSize * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_ansY, d_newAnsY, newAnsSize * sizeof(float), cudaMemcpyDeviceToDevice);
        
        ansSize = newAnsSize;

        // If no points remaining
        if (newN == 0) {
            break;
        }
        
        computeLocalScatterIdx<<<numBlocks, BLOCK_SIZE>>>(
            d_newLabels, d_keepFlags, d_labelOffsets, d_scatterIdx, d_labelCounters, currentN);
        
        compactPointsKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_newLabels,
            d_px_new, d_py_new, d_labels_new,
            d_keepFlags, d_scatterIdx, currentN);
        
        d_px = d_px_new;
        d_py = d_py_new;
        d_labels = d_labels_new;
        
        currentN = newN;
        numLabels = newNumLabels;
    }

    h_ansX = (float*)malloc(ansSize * sizeof(float));
    h_ansY = (float*)malloc(ansSize * sizeof(float));
    cudaMemcpy(h_ansX, d_ansX, ansSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ansY, d_ansY, ansSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 1; i < ansSize - 1; i++) {
        hullPoints.push_back({h_ansX[i], h_ansY[i]});
    }
    
    free(h_ansX);
    free(h_ansY);

    cudaFree(d_state);
    cudaFree(d_goesLeft);
    cudaFree(d_goesRight);
    cudaFree(d_segmentOffsets);
    cudaFree(d_maxPerSegment);
    cudaFree(d_statePrefixSum);
    cudaFree(d_newLabels);
    cudaFree(d_keepFlags);
    cudaFree(d_scatterIdx);
    cudaFree(d_labelCounts);
    cudaFree(d_labelOffsets);
    cudaFree(d_labelCounters);
    cudaFree(d_newAnsX);
    cudaFree(d_newAnsY);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_labels);
    cudaFree(d_px_new);
    cudaFree(d_py_new);
    cudaFree(d_labels_new);
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_distances);
    cudaFree(d_labels);
    cudaFree(d_ansX);
    cudaFree(d_ansY);
}


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
