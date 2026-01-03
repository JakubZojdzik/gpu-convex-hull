#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include "utils.h"

#define BLOCK_SIZE 512

// ============================================================================
// Simple QuickHull for ONE side of the hull (upper or lower)
// All points have label=0, single partition that recursively splits
// ============================================================================

// Compute perpendicular distance from line (lx,ly)->(rx,ry) for each point
// Positive distance = point is to the LEFT of the directed line
__global__ void computeDistancesSimpleKernel(float *px, float *py,
                                              float lx, float ly, float rx, float ry,
                                              float *distances, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float curX = px[idx];
    float curY = py[idx];

    // Cross product: (r - l) x (cur - l)
    float d = (rx - lx) * (curY - ly) - (ry - ly) * (curX - lx);
    distances[idx] = d;
}

// Find index of point with maximum distance (only among positive distances)
// Returns -1 if no point has positive distance
__global__ void findMaxDistPointKernel(float *distances, int *maxIdx, float *maxDist, int n) {
    __shared__ float sharedDist[BLOCK_SIZE];
    __shared__ int sharedIdx[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize with invalid values
    sharedDist[tid] = -FLT_MAX;
    sharedIdx[tid] = -1;

    if (idx < n && distances[idx] > 0) {
        sharedDist[tid] = distances[idx];
        sharedIdx[tid] = idx;
    }
    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedDist[tid + stride] > sharedDist[tid]) {
                sharedDist[tid] = sharedDist[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Block winner updates global max atomically
    if (tid == 0 && sharedDist[0] > 0) {
        // Use atomicMax on a float by casting to int (works for positive floats)
        int oldVal = atomicMax((int*)maxDist, __float_as_int(sharedDist[0]));
        if (__float_as_int(sharedDist[0]) > oldVal) {
            *maxIdx = sharedIdx[0];
        }
    }
}

// Mark points that should go to left partition (positive distance, x < maxPoint.x)
// or right partition (positive distance, x >= maxPoint.x)
__global__ void classifyPointsSimpleKernel(float *px, float *distances, float maxPx,
                                            int *goesLeft, int *goesRight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (distances[idx] > 0) {
        if (px[idx] < maxPx) {
            goesLeft[idx] = 1;
            goesRight[idx] = 0;
        } else {
            goesLeft[idx] = 0;
            goesRight[idx] = 1;
        }
    } else {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
    }
}

// Compact points into new arrays based on prefix sums
__global__ void compactPointsKernel(float *px, float *py, float *distances,
                                     int *goesLeft, int *goesRight,
                                     int *leftScan, int *rightScan, int rightOffset,
                                     float *pxNew, float *pyNew, float *distNew, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (goesLeft[idx]) {
        int newIdx = leftScan[idx];
        pxNew[newIdx] = px[idx];
        pyNew[newIdx] = py[idx];
        distNew[newIdx] = distances[idx];
    } else if (goesRight[idx]) {
        int newIdx = rightOffset + rightScan[idx];
        pxNew[newIdx] = px[idx];
        pyNew[newIdx] = py[idx];
        distNew[newIdx] = distances[idx];
    }
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

// CUB find min/max X coordinates
void findMinMaxX(float *d_px, int n, float *minX, float *maxX) {
    float *d_result;
    cudaMalloc(&d_result, sizeof(float));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Min
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_px, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_px, d_result, n);
    cudaMemcpy(minX, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);

    // Max
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_px, d_result, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_px, d_result, n);
    cudaMemcpy(maxX, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_temp_storage);

    cudaFree(d_result);
}

// ============================================================================
// QuickHull for one side: finds hull points between leftPt and rightPt
// Points are assumed to be on the LEFT side of the directed edge leftPt->rightPt
// Returns hull points in order from leftPt to rightPt (exclusive of endpoints)
// ============================================================================
void gpuQuickHullOneSide(float *h_px, float *h_py, int n,
                          float leftX, float leftY, float rightX, float rightY,
                          std::vector<Point> &hullPoints) {
    if (n == 0) return;

    // Allocate device memory
    float *d_px, *d_py, *d_pxNew, *d_pyNew;
    float *d_distances, *d_distNew;
    int *d_goesLeft, *d_goesRight, *d_leftScan, *d_rightScan;

    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMalloc(&d_pxNew, n * sizeof(float));
    cudaMalloc(&d_pyNew, n * sizeof(float));
    cudaMalloc(&d_distances, n * sizeof(float));
    cudaMalloc(&d_distNew, n * sizeof(float));
    cudaMalloc(&d_goesLeft, n * sizeof(int));
    cudaMalloc(&d_goesRight, n * sizeof(int));
    cudaMalloc(&d_leftScan, n * sizeof(int));
    cudaMalloc(&d_rightScan, n * sizeof(int));

    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);

    // ANS stores the hull points in order
    std::vector<Point> ans;
    ans.push_back({leftX, leftY});
    ans.push_back({rightX, rightY});

    // Partition info: each partition is defined by consecutive ANS points
    // partition i goes from ans[i] to ans[i+1]
    // We store: start index in point array, count of points
    struct Partition {
        int start;
        int count;
    };
    std::vector<Partition> partitions;
    partitions.push_back({0, n});

    int currentN = n;
    float *h_pxTemp = new float[n];
    float *h_pyTemp = new float[n];

    while (true) {
        bool anyChanged = false;

        // For each partition, find max point and split
        std::vector<Point> newAns;
        std::vector<Partition> newPartitions;
        
        // We'll process all partitions and rebuild arrays
        std::vector<float> allNewPx, allNewPy;
        
        int partIdx = 0;
        for (size_t p = 0; p < partitions.size(); p++) {
            Partition &part = partitions[p];
            Point &L = ans[p];
            Point &R = ans[p + 1];

            newAns.push_back(L);

            if (part.count == 0) {
                // Empty partition, no max point
                continue;
            }

            // Compute distances for this partition's points
            int numBlocks = (part.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            computeDistancesSimpleKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px + part.start, d_py + part.start,
                L.x, L.y, R.x, R.y,
                d_distances + part.start, part.count);
            cudaDeviceSynchronize();

            // Find max distance point
            int h_maxIdx = -1;
            float h_maxDist = -FLT_MAX;
            int *d_maxIdx;
            float *d_maxDist;
            cudaMalloc(&d_maxIdx, sizeof(int));
            cudaMalloc(&d_maxDist, sizeof(float));
            cudaMemcpy(d_maxIdx, &h_maxIdx, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_maxDist, &h_maxDist, sizeof(float), cudaMemcpyHostToDevice);

            findMaxDistPointKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_distances + part.start, d_maxIdx, d_maxDist, part.count);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_maxIdx, d_maxIdx, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_maxDist, d_maxDist, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_maxIdx);
            cudaFree(d_maxDist);

            if (h_maxIdx < 0 || h_maxDist <= 0) {
                // No point outside the line, partition is done
                continue;
            }

            anyChanged = true;

            // Get max point coordinates
            float maxPx, maxPy;
            cudaMemcpy(&maxPx, d_px + part.start + h_maxIdx, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&maxPy, d_py + part.start + h_maxIdx, sizeof(float), cudaMemcpyDeviceToHost);

            // Add max point to ANS (between L and R)
            newAns.push_back({maxPx, maxPy});

            // Classify points: left of maxPoint or right of maxPoint
            // Left partition: points with positive distance from L->maxP
            // Right partition: points with positive distance from maxP->R
            
            // For simplicity, we use X coordinate to split (as in paper)
            // Points with x < maxPx go to left partition, x >= maxPx go to right
            classifyPointsSimpleKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px + part.start, d_distances + part.start, maxPx,
                d_goesLeft + part.start, d_goesRight + part.start, part.count);
            cudaDeviceSynchronize();

            // Prefix sums for compaction
            cubExclusiveScanInt(d_goesLeft + part.start, d_leftScan + part.start, part.count);
            cubExclusiveScanInt(d_goesRight + part.start, d_rightScan + part.start, part.count);

            // Get counts
            int leftCount, rightCount;
            int lastLeft, lastRight;
            cudaMemcpy(&leftCount, d_leftScan + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastLeft, d_goesLeft + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            leftCount += lastLeft;
            cudaMemcpy(&rightCount, d_rightScan + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastRight, d_goesRight + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            rightCount += lastRight;

            // Compact points
            int newStart = allNewPx.size();
            allNewPx.resize(newStart + leftCount + rightCount);
            allNewPy.resize(newStart + leftCount + rightCount);

            // Copy to device temp arrays
            compactPointsKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px + part.start, d_py + part.start, d_distances + part.start,
                d_goesLeft + part.start, d_goesRight + part.start,
                d_leftScan + part.start, d_rightScan + part.start, leftCount,
                d_pxNew, d_pyNew, d_distNew, part.count);
            cudaDeviceSynchronize();

            // Copy compacted points back to host temp
            cudaMemcpy(h_pxTemp, d_pxNew, (leftCount + rightCount) * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_pyTemp, d_pyNew, (leftCount + rightCount) * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < leftCount + rightCount; i++) {
                allNewPx[newStart + i] = h_pxTemp[i];
                allNewPy[newStart + i] = h_pyTemp[i];
            }

            // Record new partitions
            newPartitions.push_back({newStart, leftCount});  // L -> maxP
            newPartitions.push_back({newStart + leftCount, rightCount});  // maxP -> R
        }

        // Add final ANS point
        newAns.push_back(ans.back());

        if (!anyChanged) {
            // No more splits, we're done
            break;
        }

        // Update for next iteration
        ans = newAns;
        partitions = newPartitions;

        // Copy new points to device
        currentN = allNewPx.size();
        if (currentN > 0) {
            cudaMemcpy(d_px, allNewPx.data(), currentN * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_py, allNewPy.data(), currentN * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    delete[] h_pxTemp;
    delete[] h_pyTemp;

    // Cleanup
    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pxNew);
    cudaFree(d_pyNew);
    cudaFree(d_distances);
    cudaFree(d_distNew);
    cudaFree(d_goesLeft);
    cudaFree(d_goesRight);
    cudaFree(d_leftScan);
    cudaFree(d_rightScan);

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

    // Find min and max X points
    float minX = FLT_MAX, maxX = -FLT_MAX;
    int minIdx = 0, maxIdx = 0;
    for (int i = 0; i < n; i++) {
        if (h_px[i] < minX || (h_px[i] == minX && h_py[i] < h_py[minIdx])) {
            minX = h_px[i];
            minIdx = i;
        }
        if (h_px[i] > maxX || (h_px[i] == maxX && h_py[i] > h_py[maxIdx])) {
            maxX = h_px[i];
            maxIdx = i;
        }
    }

    Point minPt = {h_px[minIdx], h_py[minIdx]};
    Point maxPt = {h_px[maxIdx], h_py[maxIdx]};

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
