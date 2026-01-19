#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846f

// ================= Algorithms =================

// CPU
extern void cpuMonotoneChain(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

extern void cpuQuickHull(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

// GPU
extern "C" void gpuQuickHull(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

extern "C" void gpuQuickHullnaive(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

// ================= Point generation =================

__global__ void generate_points(float *px, float *py, int N, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float u = curand_uniform(&state);
    float theta = curand_uniform(&state) * 2.0f * PI;

    float r = sqrtf(u);
    px[idx] = r * cosf(theta);
    py[idx] = r * sinf(theta);
}

// ================= Timing helper =================

double elapsed_ms(
    std::chrono::high_resolution_clock::time_point a,
    std::chrono::high_resolution_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// ================= Main =================

int main()
{
    const std::vector<int> input_sizes = {
        1000000,
        5000000,
        10000000,
        50000000,
        100000000,
        500000000
    };

    const int NUM_SETS  = 10;
    const int NUM_RUNS  = 3;

    for (int N : input_sizes) {

        printf("\n=====================================================\n");
        printf("N = %d\n", N);
        printf("=====================================================\n");

        // Allocate GPU memory for points
        float *d_px, *d_py;
        cudaMalloc(&d_px, N * sizeof(float));
        cudaMalloc(&d_py, N * sizeof(float));

        // Allocate host memory
        float *h_px = (float*)malloc(N * sizeof(float));
        float *h_py = (float*)malloc(N * sizeof(float));
        float *result_x = (float*)malloc(N * sizeof(float));
        float *result_y = (float*)malloc(N * sizeof(float));

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        for (int set = 0; set < NUM_SETS; ++set) {

            printf("\n--- Input set %d / %d ---\n", set + 1, NUM_SETS);

            // Generate new input set
            unsigned long long seed = 123456ULL + set;
            generate_points<<<blocks, threads>>>(d_px, d_py, N, seed);
            cudaDeviceSynchronize();

            cudaMemcpy(h_px, d_px, N * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_py, d_py, N * sizeof(float), cudaMemcpyDeviceToHost);

            /* ================= CPU Monotone Chain ================= */

            std::vector<double> cpu_times;
            int M_cpu = 0;

            for (int run = 0; run < NUM_RUNS; ++run) {
                memset(result_x, 0, N * sizeof(float));
                memset(result_y, 0, N * sizeof(float));

                auto t0 = std::chrono::high_resolution_clock::now();
                cpuMonotoneChain(h_px, h_py, N, result_x, result_y, &M_cpu);
                auto t1 = std::chrono::high_resolution_clock::now();

                cpu_times.push_back(elapsed_ms(t0, t1));
            }

            printf("CPU Monotone Chain:\n");
            printf("  Hull size: %d\n", M_cpu);
            printf("  First run: %.3f ms\n", cpu_times[0]);
            printf("  All runs : ");
            for (double t : cpu_times) printf("%.3f ms  ", t);
            printf("\n");

            /* ================= GPU QuickHull ================= */

            std::vector<double> gpu_times;
            int M_gpu = 0;

            for (int run = 0; run < NUM_RUNS; ++run) {
                memset(result_x, 0, N * sizeof(float));
                memset(result_y, 0, N * sizeof(float));

                auto t0 = std::chrono::high_resolution_clock::now();
                gpuQuickHull(h_px, h_py, N, result_x, result_y, &M_gpu);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();

                gpu_times.push_back(elapsed_ms(t0, t1));
            }

            printf("GPU QuickHull:\n");
            printf("  Hull size: %d\n", M_gpu);
            printf("  First run: %.3f ms\n", gpu_times[0]);
            printf("  All runs : ");
            for (double t : gpu_times) printf("%.3f ms  ", t);
            printf("\n");

            /* ================= GPU Naive QuickHull ================= */

            std::vector<double> gpu_naive_times;

            for (int run = 0; run < NUM_RUNS; ++run) {
                memset(result_x, 0, N * sizeof(float));
                memset(result_y, 0, N * sizeof(float));

                auto t0 = std::chrono::high_resolution_clock::now();
                gpuQuickHullnaive(h_px, h_py, N, result_x, result_y, &M_gpu);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();

                gpu_naive_times.push_back(elapsed_ms(t0, t1));
            }

            printf("GPU Naive QuickHull:\n");
            printf("  Hull size: %d\n", M_gpu);
            printf("  First run: %.3f ms\n", gpu_naive_times[0]);
            printf("  All runs : ");
            for (double t : gpu_naive_times) printf("%.3f ms  ", t);
            printf("\n");
        }

        free(h_px);
        free(h_py);
        free(result_x);
        free(result_y);
        cudaFree(d_px);
        cudaFree(d_py);
    }

    return 0;
}
