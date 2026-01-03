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
    int N = 700000;

    float *px = (float*) malloc(sizeof(float) * N);
    float *py = (float*) malloc(sizeof(float) * N);

    srand(time(NULL));

    int points = 0;
    float radius = 50.0f;
    while (points < N) {
        float x = radius * 2.0f * (rand() / (float)RAND_MAX) - radius;
        float y = radius * 2.0f * (rand() / (float)RAND_MAX) - radius;
        if (x*x + y*y <= radius*radius) {
            px[points] = x;
            py[points] = y;
            points++;
        }
    }

    // printf("Input Points:\n[");
    // for (int i = 0; i < N; i++) {
    //     printf("(%.3f, %.3f)", px[i], py[i]);
    //     if (i < N - 1) printf(", ");
    // }
    // printf("]\n");

    float *result_x = (float*) malloc(sizeof(float) * N);
    float *result_y = (float*) malloc(sizeof(float) * N);
    int M_cpu = 0, M_gpu = 0;

    /* ================= CPU ================= */
    auto cpu_start = std::chrono::high_resolution_clock::now();
    grahamScan(px, py, N, result_x, result_y, &M_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    printf("CPU Graham Scan:\n[");
    for (int i = 0; i < M_cpu; i++) {
        printf("(%.3f, %.3f)", result_x[i], result_y[i]);
        if (i < M_cpu - 1) printf(", ");
    }
    printf("]\n");
    printf("Hull size: %d\n", M_cpu);
    printf("Time: %.3f ms\n\n", cpu_ms);

    /* ================= GPU ================= */
    // Timed run
    memset(result_x, 0, sizeof(float) * N);
    memset(result_y, 0, sizeof(float) * N);
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpuQuickHull(px, py, N, result_x, result_y, &M_gpu);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    printf("GPU QuickHull:\n[");
    for (int i = 0; i < M_gpu; i++) {
        printf("(%.3f, %.3f)", result_x[i], result_y[i]);
        if (i < M_gpu - 1) printf(", ");
    }
    printf("]\n");
    printf("Hull size: %d\n", M_gpu);
    printf("Time: %.3f ms\n", gpu_ms);

    free(px);
    free(py);
    free(result_x);
    free(result_y);

    return 0;
}
