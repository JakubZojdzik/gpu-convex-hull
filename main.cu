#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846f

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

// Visualizer
extern "C" void visualizeConvexHull(float* input_x, float* input_y, int num_points,
                                     float* hull_x, float* hull_y, int hull_size,
                                     const char* output_filename);


__global__ void generate_points(float *px, float *py, int N, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    // Generate uniform random numbers
    float u = curand_uniform(&state);        // [0,1)
    float theta = curand_uniform(&state) * 2.0f * PI;

    // Map to circle
    float r = sqrtf(u);
    px[idx] = r * cosf(theta);
    py[idx] = r * sinf(theta);
}

int main()
{
    int N = 100000000;

    float *d_px, *d_py;
    cudaMalloc(&d_px, N * sizeof(float));
    cudaMalloc(&d_py, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    generate_points<<<blocks, threads>>>(d_px, d_py, N, 123456ULL);
    cudaDeviceSynchronize();
    float *h_px = (float*)malloc(N*sizeof(float));
    float *h_py = (float*)malloc(N*sizeof(float));
    cudaMemcpy(h_px, d_px, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_py, d_py, N*sizeof(float), cudaMemcpyDeviceToHost);

    float *result_x = (float*) malloc(sizeof(float) * N);
    float *result_y = (float*) malloc(sizeof(float) * N);
    int M_cpu = 0, M_gpu = 0;

    /* ================= CPU ================= */
    // auto cpu_start = std::chrono::high_resolution_clock::now();
    // cpuMonotoneChain(px, py, N, result_x, result_y, &M_cpu);
    // auto cpu_end = std::chrono::high_resolution_clock::now();
    // double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // printf("CPU Monotone chain:\n");
    // printf("[");
    // for (int i = 0; i < M_cpu; i++) {
    //     printf("(%.3f, %.3f)", result_x[i], result_y[i]);
    //     if (i < M_cpu - 1) printf(", ");
    // }
    // printf("]\n");
    // printf("Hull size: %d\n", M_cpu);
    // printf("Time: %.3f ms\n\n", cpu_ms);


    /* ================= GPU ================= */
    // Timed run
    memset(result_x, 0, sizeof(float) * N);
    memset(result_y, 0, sizeof(float) * N);
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpuQuickHull(h_px, h_py, N, result_x, result_y, &M_gpu);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    printf("GPU QuickHull:\n");
    // printf("[");
    // for (int i = 0; i < M_gpu; i++) {
    //     printf("(%.3f, %.3f)", result_x[i], result_y[i]);
    //     if (i < M_gpu - 1) printf(", ");
    // }
    // printf("]\n");
    printf("Hull size: %d\n", M_gpu);
    printf("Time: %.3f ms\n\n", gpu_ms);

   
    /* ================= GPU naive ================= */
    // Timed run
    memset(result_x, 0, sizeof(float) * N);
    memset(result_y, 0, sizeof(float) * N);
    gpu_start = std::chrono::high_resolution_clock::now();
    gpuQuickHullnaive(h_px, h_py, N, result_x, result_y, &M_gpu);
    cudaDeviceSynchronize();
    gpu_end = std::chrono::high_resolution_clock::now();
    gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    printf("GPU naive QuickHull:\n");
    // printf("[");
    // for (int i = 0; i < M_gpu; i++) {
    //     printf("(%.3f, %.3f)", result_x[i], result_y[i]);
    //     if (i < M_gpu - 1) printf(", ");
    // }
    // printf("]\n");
    printf("Hull size: %d\n", M_gpu);
    printf("Time: %.3f ms\n\n", gpu_ms);

    // Visualize the result
    // printf("Creating visualization...\n");
    // visualizeConvexHull(px, py, N, result_x, result_y, M_gpu, "convex_hull_visualization");

    free(h_px);
    free(h_py);
    free(result_x);
    free(result_y);
    cudaFree(d_px);
    cudaFree(d_py);

    return 0;
}