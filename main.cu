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
    // float *px = new float[N]{
    //     5861.843f, -7883.658f, -7156.774f, -7887.822f, 3757.629f,
    //     2383.592f, 5502.341f, -6789.278f, 2519.980f, -2043.839f,
    //     5122.345f, 2577.598f, 1792.777f, -5110.984f, -2623.934f,
    //     -8583.311f, 7381.590f, 3621.583f, 660.616f, -6811.547f,
    //     1278.297f, 459.991f, 6219.344f, -1606.287f, 3004.084f,
    //     6903.475f, -3202.690f, 1339.803f, -5182.649f, 4080.470f
    // };

    // float *py = new float[N]{
    //     966.890f, -5072.616f, -2101.514f, -1629.504f, 6149.915f,
    //     -8782.118f, -1996.346f, 6702.553f, 4663.176f, -9240.954f,
    //     3457.333f, 4789.114f, 4693.940f, -6841.067f, -4733.226f,
    //     -622.064f, -4437.701f, -98.430f, 7820.965f, -4217.040f,
    //     5766.052f, -9711.430f, -611.857f, -2262.690f, 7419.375f,
    //     4420.772f, 9255.053f, -5821.100f, -5038.614f, -4957.175f
    // };

    // (-6789.278, 6702.553)
    // (3757.629, 6149.915)
    // (-3202.690, 9255.053)
    // (6903.475, 4420.772)
    // (3004.084, 7419.375)
    int N = 5;
    float *px = new float[N]{
        -6789.278f, 3757.629f, -3202.690f, 6903.475f, 3004.084f
    };
    float *py = new float[N]{
        6702.553f, 6149.915f, 9255.053f, 4420.772f, 7419.375f
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