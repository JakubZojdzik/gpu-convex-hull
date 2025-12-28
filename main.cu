#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// CPU
extern void grahamScan(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

// GPU
extern "C" void gpuQuickHull(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

int main()
{
    int N = 100000;

    float *px = (float*) malloc(sizeof(float) * N);
    float *py = (float*) malloc(sizeof(float) * N);

    srand(123);

    int points = 0;
    while (points < N) {
        float x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        float y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        if (x*x + y*y <= 1.0f) {
            px[points] = x;
            py[points] = y;
            points++;
        }
    }

    float *result_x = (float*) malloc(sizeof(float) * N);
    float *result_y = (float*) malloc(sizeof(float) * N);
    int M_cpu = 0, M_gpu = 0;

    /* ================= CPU ================= */
    auto cpu_start = std::chrono::high_resolution_clock::now();
    grahamScan(px, py, N, result_x, result_y, &M_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    printf("CPU Graham Scan\n");
    printf("Hull size: %d\n", M_cpu);
    printf("Time: %.3f ms\n\n", cpu_ms);

    /* ================= GPU ================= */
    // Warm-up run
    gpuQuickHull(px, py, N, result_x, result_y, &M_gpu);
    cudaDeviceSynchronize();
    
    // Timed run
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpuQuickHull(px, py, N, result_x, result_y, &M_gpu);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    printf("GPU QuickHull\n");
    printf("Hull size: %d\n", M_gpu);
    printf("Time: %.3f ms\n", gpu_ms);

    free(px);
    free(py);
    free(result_x);
    free(result_y);

    return 0;
}
