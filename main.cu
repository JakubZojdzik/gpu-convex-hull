#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

// CPU
extern void monotoneChain(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

extern void grahamScan(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

// GPU
extern "C" void gpuQuickHull(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

static int countUniquePoints(const float *x,
                             const float *y,
                             int n,
                             float eps = 1e-6f)
{
    if (n == 0) return 0;

    int unique = 0;

    for (int i = 0; i < n; i++) {
        bool is_new = true;
        for (int j = 0; j < i; j++) {
            if (fabs(x[i] - x[j]) <= eps &&
                fabs(y[i] - y[j]) <= eps)
            {
                is_new = false;
                break;
            }
        }
        if (is_new) unique++;
    }
    return unique;
}

int main()
{
    // int N = 30;

    // float *px = (float*) malloc(sizeof(float) * N);
    // float *py = (float*) malloc(sizeof(float) * N);

    // srand(time(NULL));

    // int points = 0;
    // float radius = 10000.0f;
    // while (points < N) {
    //     float x = radius * 2.0f * (rand() / (float)RAND_MAX) - radius;
    //     float y = radius * 2.0f * (rand() / (float)RAND_MAX) - radius;
    //     if (x*x + y*y <= radius*radius) {
    //         px[points] = x;
    //         py[points] = y;
    //         points++;
    //     }
    // }

    // input points:
    // [(-4784.123, 6416.947), (-7254.780, 390.430), (1759.670, -9330.546), (-1073.548, 944.183), (-5673.735, -1845.406), (8004.467, -1873.345), (2795.154, 1064.468), (-5812.456, 6804.494), (8167.760, 30.967), (-2368.432, -4627.420), (1811.053, -443.390), (-251.239, -5228.379)]


    int N = 12;
    float *px = new float[N]{
        -4784.123f,  -7254.780f,  1759.670f,  -1073.548f,
        -5673.735f,  8004.467f,   2795.154f,  -5812.456f,
        8167.760f,   -2368.432f,  1811.053f,  -251.239f
    };
    float *py = new float[N]{
        6416.947f,   390.430f,    -9330.546f, 944.183f,
        -1845.406f,  -1873.345f,  1064.468f,  6804.494f,
        30.967f,     -4627.420f,  -443.390f,  -5228.379f
    };

    printf("Input Points:\n");
    printf("[");
    for (int i = 0; i < N; i++) {
        printf("(%.3f, %.3f)", px[i], py[i]);
        if (i < N - 1) printf(", ");
    }
    printf("]\n");

    float *result_x = (float*) malloc(sizeof(float) * N);
    float *result_y = (float*) malloc(sizeof(float) * N);
    int M_cpu = 0, M_gpu = 0;

    /* ================= CPU ================= */
    auto cpu_start = std::chrono::high_resolution_clock::now();
    monotoneChain(px, py, N, result_x, result_y, &M_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    printf("CPU Monotone chain:\n");
    printf("[");
    for (int i = 0; i < M_cpu; i++) {
        printf("(%.3f, %.3f)", result_x[i], result_y[i]);
        if (i < M_cpu - 1) printf(", ");
    }
    printf("]\n");
    int unique_cpu = countUniquePoints(result_x, result_y, M_cpu);
    printf("Unique hull points: %d\n", unique_cpu);
    printf("Hull size: %d\n", M_cpu);
    printf("Time: %.3f ms\n", cpu_ms);

//    cpu_start = std::chrono::high_resolution_clock::now();
//    grahamScan(px, py, N, result_x, result_y, &M_cpu);
//    cpu_end = std::chrono::high_resolution_clock::now();
//    cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
//
//    printf("CPU Graham scan:\n[");
//    for (int i = 0; i < M_cpu; i++) {
//        printf("(%.3f, %.3f)", result_x[i], result_y[i]);
//        if (i < M_cpu - 1) printf(", ");
//    }
//    printf("]\n");
//    printf("Hull size: %d\n", M_cpu);
//    printf("Time: %.3f ms\n\n", cpu_ms);

    /* ================= GPU ================= */
    // Timed run
    memset(result_x, 0, sizeof(float) * N);
    memset(result_y, 0, sizeof(float) * N);
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpuQuickHull(px, py, N, result_x, result_y, &M_gpu);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    printf("GPU QuickHull:\n");
    printf("[");
    for (int i = 0; i < M_gpu; i++) {
        printf("(%.3f, %.3f)", result_x[i], result_y[i]);
        if (i < M_gpu - 1) printf(", ");
    }
    printf("]\n");
    int unique_gpu = countUniquePoints(result_x, result_y, M_gpu);
    printf("Unique hull points: %d\n", unique_gpu);
    printf("Hull size: %d\n", M_gpu);
    printf("Time: %.3f ms\n", gpu_ms);

    free(px);
    free(py);
    free(result_x);
    free(result_y);

    return 0;
}