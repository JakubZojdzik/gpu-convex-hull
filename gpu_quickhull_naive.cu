#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <climits>
#include <vector>
#include <algorithm>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include "utils.h"

#define BLOCK_SIZE 128


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



__global__ void computeDistancesSimpleKernel(float *px, float *py,
                                              float lx, float ly, float rx, float ry,
                                              float *distances, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float curX = px[idx];
    float curY = py[idx];

    float d = (rx - lx) * (curY - ly) - (ry - ly) * (curX - lx);
    distances[idx] = d;
}

__global__ void findMaxDistPointKernel(float *distances, float *blockMaxDist, int *blockMaxIdx, int n) {
    __shared__ float sharedDist[BLOCK_SIZE];
    __shared__ int sharedIdx[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedDist[tid] = -FLT_MAX;
    sharedIdx[tid] = -1;

    if (idx < n && distances[idx] > 0) {
        sharedDist[tid] = distances[idx];
        sharedIdx[tid] = idx;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sharedDist[tid + stride] > sharedDist[tid]) {
                sharedDist[tid] = sharedDist[tid + stride];
                sharedIdx[tid] = sharedIdx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMaxDist[blockIdx.x] = sharedDist[0];
        blockMaxIdx[blockIdx.x] = sharedIdx[0];
    }
}

void findMaxPointHost(float *h_blockMaxDist, int *h_blockMaxIdx, int numBlocks, 
                      int *outIdx, float *outDist) {
    *outIdx = -1;
    *outDist = -FLT_MAX;
    for (int i = 0; i < numBlocks; i++) {
        if (h_blockMaxDist[i] > *outDist) {
            *outDist = h_blockMaxDist[i];
            *outIdx = h_blockMaxIdx[i];
        }
    }
}

__global__ void classifyPointsForSplitKernel(float *px, float *py, float *oldDistances,
                                              float lx, float ly, float mx, float my, float rx, float ry,
                                              int maxIdx,
                                              int *goesLeft, int *goesRight,
                                              float *newDistLeft, float *newDistRight, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (idx == maxIdx) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        newDistLeft[idx] = 0;
        newDistRight[idx] = 0;
        return;
    }

    if (oldDistances[idx] <= 0) {
        goesLeft[idx] = 0;
        goesRight[idx] = 0;
        newDistLeft[idx] = 0;
        newDistRight[idx] = 0;
        return;
    }

    float curX = px[idx];
    float curY = py[idx];

    float distLM = (mx - lx) * (curY - ly) - (my - ly) * (curX - lx);
    float distMR = (rx - mx) * (curY - my) - (ry - my) * (curX - mx);

    goesLeft[idx] = (distLM > 0) ? 1 : 0;
    goesRight[idx] = (distMR > 0) ? 1 : 0;
    
    newDistLeft[idx] = distLM;
    newDistRight[idx] = distMR;
}

__global__ void compactPointsKernel(float *px, float *py,
                                     float *newDistLeft, float *newDistRight,
                                     int *goesLeft, int *goesRight,
                                     int *leftScan, int *rightScan, int rightOffset,
                                     float *pxNew, float *pyNew, float *distNew, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (goesLeft[idx]) {
        int newIdx = leftScan[idx];
        pxNew[newIdx] = px[idx];
        pyNew[newIdx] = py[idx];
        distNew[newIdx] = newDistLeft[idx];
    } else if (goesRight[idx]) {
        int newIdx = rightOffset + rightScan[idx];
        pxNew[newIdx] = px[idx];
        pyNew[newIdx] = py[idx];
        distNew[newIdx] = newDistRight[idx];
    }
}

static void cubExclusiveScanInt(int *d_input, int *d_output, int n) {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output, n);
    cudaFree(d_temp_storage);
}


static void gpuQuickHullOneSide(float *h_px, float *h_py, int n,
                          float leftX, float leftY, float rightX, float rightY,
                          std::vector<Point> &hullPoints) {
    if (n == 0) return;

    float *d_px, *d_py, *d_pxNew, *d_pyNew;
    float *d_distances, *d_distNew;
    float *d_newDistLeft, *d_newDistRight;
    int *d_goesLeft, *d_goesRight, *d_leftScan, *d_rightScan;

    cudaMalloc(&d_px, n * sizeof(float));
    cudaMalloc(&d_py, n * sizeof(float));
    cudaMalloc(&d_pxNew, n * sizeof(float));
    cudaMalloc(&d_pyNew, n * sizeof(float));
    cudaMalloc(&d_distances, n * sizeof(float));
    cudaMalloc(&d_distNew, n * sizeof(float));
    cudaMalloc(&d_newDistLeft, n * sizeof(float));
    cudaMalloc(&d_newDistRight, n * sizeof(float));
    cudaMalloc(&d_goesLeft, n * sizeof(int));
    cudaMalloc(&d_goesRight, n * sizeof(int));
    cudaMalloc(&d_leftScan, n * sizeof(int));
    cudaMalloc(&d_rightScan, n * sizeof(int));

    cudaMemcpy(d_px, h_px, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<Point> ans;
    ans.push_back({leftX, leftY});
    ans.push_back({rightX, rightY});

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

        std::vector<Point> newAns;
        std::vector<Partition> newPartitions;
        std::vector<float> allNewPx, allNewPy;
        
        for (size_t p = 0; p < partitions.size(); p++) {
            Partition &part = partitions[p];
            Point &L = ans[p];
            Point &R = ans[p + 1];

            newAns.push_back(L);

            if (part.count == 0) {
                newPartitions.push_back({(int)allNewPx.size(), 0});
                continue;
            }

            int numBlocks = (part.count + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            computeDistancesSimpleKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px + part.start, d_py + part.start,
                L.x, L.y, R.x, R.y,
                d_distances + part.start, part.count);
            cudaDeviceSynchronize();

            float *d_blockMaxDist;
            int *d_blockMaxIdx;
            cudaMalloc(&d_blockMaxDist, numBlocks * sizeof(float));
            cudaMalloc(&d_blockMaxIdx, numBlocks * sizeof(int));

            findMaxDistPointKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_distances + part.start, d_blockMaxDist, d_blockMaxIdx, part.count);
            cudaDeviceSynchronize();

            float *h_blockMaxDist = new float[numBlocks];
            int *h_blockMaxIdx = new int[numBlocks];
            cudaMemcpy(h_blockMaxDist, d_blockMaxDist, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_blockMaxIdx, d_blockMaxIdx, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
            
            int h_maxIdx;
            float h_maxDist;
            findMaxPointHost(h_blockMaxDist, h_blockMaxIdx, numBlocks, &h_maxIdx, &h_maxDist);
            
            delete[] h_blockMaxDist;
            delete[] h_blockMaxIdx;
            cudaFree(d_blockMaxDist);
            cudaFree(d_blockMaxIdx);

            if (h_maxIdx < 0 || h_maxDist <= 0) {
                newPartitions.push_back({(int)allNewPx.size(), 0});
                continue;
            }

            anyChanged = true;

            float maxPx, maxPy;
            cudaMemcpy(&maxPx, d_px + part.start + h_maxIdx, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&maxPy, d_py + part.start + h_maxIdx, sizeof(float), cudaMemcpyDeviceToHost);

            newAns.push_back({maxPx, maxPy});

            classifyPointsForSplitKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px + part.start, d_py + part.start, d_distances + part.start,
                L.x, L.y, maxPx, maxPy, R.x, R.y,
                h_maxIdx,
                d_goesLeft + part.start, d_goesRight + part.start,
                d_newDistLeft + part.start, d_newDistRight + part.start, part.count);
            cudaDeviceSynchronize();

            cubExclusiveScanInt(d_goesLeft + part.start, d_leftScan + part.start, part.count);
            cubExclusiveScanInt(d_goesRight + part.start, d_rightScan + part.start, part.count);

            int leftCount, rightCount;
            int lastLeft, lastRight;
            cudaMemcpy(&leftCount, d_leftScan + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastLeft, d_goesLeft + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            leftCount += lastLeft;
            cudaMemcpy(&rightCount, d_rightScan + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastRight, d_goesRight + part.start + part.count - 1, sizeof(int), cudaMemcpyDeviceToHost);
            rightCount += lastRight;

            int newStart = allNewPx.size();
            allNewPx.resize(newStart + leftCount + rightCount);
            allNewPy.resize(newStart + leftCount + rightCount);

            compactPointsKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px + part.start, d_py + part.start,
                d_newDistLeft + part.start, d_newDistRight + part.start,
                d_goesLeft + part.start, d_goesRight + part.start,
                d_leftScan + part.start, d_rightScan + part.start, leftCount,
                d_pxNew, d_pyNew, d_distNew, part.count);
            cudaDeviceSynchronize();

            cudaMemcpy(h_pxTemp, d_pxNew, (leftCount + rightCount) * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_pyTemp, d_pyNew, (leftCount + rightCount) * sizeof(float), cudaMemcpyDeviceToHost);

            for (int i = 0; i < leftCount + rightCount; i++) {
                allNewPx[newStart + i] = h_pxTemp[i];
                allNewPy[newStart + i] = h_pyTemp[i];
            }

            newPartitions.push_back({newStart, leftCount});
            newPartitions.push_back({newStart + leftCount, rightCount});
        }

        newAns.push_back(ans.back());

        if (!anyChanged) {
            break;
        }

        ans = newAns;
        partitions = newPartitions;

        currentN = allNewPx.size();
        if (currentN > 0) {
            cudaMemcpy(d_px, allNewPx.data(), currentN * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_py, allNewPy.data(), currentN * sizeof(float), cudaMemcpyHostToDevice);
        }
    }

    delete[] h_pxTemp;
    delete[] h_pyTemp;

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_pxNew);
    cudaFree(d_pyNew);
    cudaFree(d_distances);
    cudaFree(d_distNew);
    cudaFree(d_newDistLeft);
    cudaFree(d_newDistRight);
    cudaFree(d_goesLeft);
    cudaFree(d_goesRight);
    cudaFree(d_leftScan);
    cudaFree(d_rightScan);

    for (size_t i = 1; i < ans.size() - 1; i++) {
        hullPoints.push_back(ans[i]);
    }
}

extern "C" void gpuQuickHullnaive(float *h_px, float *h_py, int n,
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
        if (d > 0) {
            upperX.push_back(h_px[i]);
            upperY.push_back(h_py[i]);
        } else if (d < 0) {
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
