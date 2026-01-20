#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <time.h>
#include <chrono>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define PI 3.14159265358979323846f

///////
extern void cpuMonotoneChain(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

extern "C" void gpuQuickHull(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);

extern "C" void gpuQuickHullNaive(
    float *p_x, float *p_y, int N,
    float *result_x, float *result_y, int *M
);
///////


extern "C" void visualizeConvexHull(
    float *points_x, float *points_y, int N,
    float *hull_x, float *hull_y, int M,
    const char *filename,
    int width, int height
);

__global__ void generate_points(float *px, float *py, int N, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    float x = curand_uniform(&state) * 2.0f - 1.0f;
    float y = curand_uniform(&state) * 2.0f - 1.0f;
    while(x*x + y*y > 1) {
        x = curand_uniform(&state);
        y = curand_uniform(&state);
    }
    px[idx] = x;
    py[idx] = y;
}


double elapsed_ms(
    std::chrono::high_resolution_clock::time_point a,
    std::chrono::high_resolution_clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}



int main()
{
    int N = 20;  // Smaller dataset for visualization
    float *d_px, *d_py;
    cudaMalloc(&d_px, N * sizeof(float));
    cudaMalloc(&d_py, N * sizeof(float));

    float *h_px = (float*)malloc(N * sizeof(float));
    float *h_py = (float*)malloc(N * sizeof(float));
    float *result_x = (float*)malloc(N * sizeof(float));
    float *result_y = (float*)malloc(N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    srand(time(NULL));
    unsigned long long seed = rand();
    generate_points<<<blocks, threads>>>(d_px, d_py, N, seed);
    cudaDeviceSynchronize();

    cudaMemcpy(h_px, d_px, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_py, d_py, N * sizeof(float), cudaMemcpyDeviceToHost);

    // cpu
    int M_cpu = 0;
    memset(result_x, 0, N * sizeof(float));
    memset(result_y, 0, N * sizeof(float));
    auto t0 = std::chrono::high_resolution_clock::now();
    cpuMonotoneChain(h_px, h_py, N, result_x, result_y, &M_cpu);
    auto t1 = std::chrono::high_resolution_clock::now();

    printf("CPU Monotone Chain:\n");
    printf("Hull size: %d\n", M_cpu);
    printf("time: %.3f ms\n", elapsed_ms(t0, t1));
    printf("\n");


    // gpu
    int M_gpu = 0;
    memset(result_x, 0, N * sizeof(float));
    memset(result_y, 0, N * sizeof(float));
    t0 = std::chrono::high_resolution_clock::now();
    gpuQuickHullNaive(h_px, h_py, N, result_x, result_y, &M_gpu);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();

    printf("GPU QuickHull Naive:\n");
    printf("Hull size: %d\n", M_gpu);
    printf("time: %.3f ms\n", elapsed_ms(t0, t1));
    printf("\n");

    // Visualize GPU result
    visualizeConvexHull(h_px, h_py, N, result_x, result_y, M_gpu, 
                              "results.ppm", 1028, 720);


    free(h_px);
    free(h_py);
    free(result_x);
    free(result_y);
    cudaFree(d_px);
    cudaFree(d_py);

    return 0;
}
