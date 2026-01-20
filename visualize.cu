#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#define MARGIN_PERCENT 0.05f  // 5% margin on each side

// Simple kernel to draw a point on the image
__global__ void drawPoints(unsigned char *image, float *px, float *py, int N, 
                           int width, int height, 
                           unsigned char r, unsigned char g, unsigned char b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    // Convert from [-1,1] coordinates to image coordinates with margin
    // First normalize from [-1,1] to [0,1], then apply margin
    float margin = MARGIN_PERCENT;
    float scale = 1.0f - 2.0f * margin;  // Scale factor accounting for margins
    
    float norm_x = (px[idx] + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    float norm_y = (py[idx] + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    
    int x = (int)((margin + norm_x * scale) * width);
    int y = (int)((margin + norm_y * scale) * height);
    
    // Bounds check
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    
    // PPM format: RGB, so 3 bytes per pixel
    int pixel_idx = (y * width + x) * 3;
    image[pixel_idx + 0] = r;
    image[pixel_idx + 1] = g;
    image[pixel_idx + 2] = b;
}

// Simple Bresenham line drawing algorithm
__device__ void drawLineSegment(unsigned char *image, int x0, int y0, int x1, int y1, 
                                int width, int height,
                                unsigned char r, unsigned char g, unsigned char b) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    
    int x = x0, y = y0;
    
    for (int i = 0; i < width + height; i++) {  // Safety limit
        if (x >= 0 && x < width && y >= 0 && y < height) {
            int pixel_idx = (y * width + x) * 3;
            image[pixel_idx + 0] = r;
            image[pixel_idx + 1] = g;
            image[pixel_idx + 2] = b;
        }
        
        if (x == x1 && y == y1) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// Kernel to draw hull lines (one thread per line segment)
__global__ void drawHullLines(unsigned char *image, float *hull_x, float *hull_y, int M,
                               int width, int height,
                               unsigned char r, unsigned char g, unsigned char b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    
    // Draw line from point idx to point (idx+1) % M
    int next_idx = (idx + 1) % M;
    
    // Convert from [-1,1] coordinates to image coordinates with margin
    float margin = MARGIN_PERCENT;
    float scale = 1.0f - 2.0f * margin;
    
    float norm_x0 = (hull_x[idx] + 1.0f) * 0.5f;
    float norm_y0 = (hull_y[idx] + 1.0f) * 0.5f;
    float norm_x1 = (hull_x[next_idx] + 1.0f) * 0.5f;
    float norm_y1 = (hull_y[next_idx] + 1.0f) * 0.5f;
    
    int x0 = (int)((margin + norm_x0 * scale) * width);
    int y0 = (int)((margin + norm_y0 * scale) * height);
    int x1 = (int)((margin + norm_x1 * scale) * width);
    int y1 = (int)((margin + norm_y1 * scale) * height);
    
    drawLineSegment(image, x0, y0, x1, y1, width, height, r, g, b);
}

// Initialize image with background color
__global__ void clearImage(unsigned char *image, int width, int height,
                           unsigned char r, unsigned char g, unsigned char b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx >= total_pixels) return;
    
    int pixel_idx = idx * 3;
    image[pixel_idx + 0] = r;
    image[pixel_idx + 1] = g;
    image[pixel_idx + 2] = b;
}

// Main visualization function
extern "C" void visualizeConvexHull(
    float *points_x, float *points_y, int N,
    float *hull_x, float *hull_y, int M,
    const char *filename,
    int width, int height)
{
    // Allocate device memory for image
    size_t image_size = width * height * 3 * sizeof(unsigned char);
    unsigned char *d_image;
    cudaMalloc(&d_image, image_size);
    
    // Clear image with white background
    int threads = 256;
    int blocks = (width * height + threads - 1) / threads;
    clearImage<<<blocks, threads>>>(d_image, width, height, 255, 255, 255);
    cudaDeviceSynchronize();
    
    // Copy points to device
    float *d_points_x, *d_points_y;
    cudaMalloc(&d_points_x, N * sizeof(float));
    cudaMalloc(&d_points_y, N * sizeof(float));
    cudaMemcpy(d_points_x, points_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_points_y, points_y, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Draw all input points in light gray
    blocks = (N + threads - 1) / threads;
    drawPoints<<<blocks, threads>>>(d_image, d_points_x, d_points_y, N, 
                                     width, height, 200, 200, 200);
    cudaDeviceSynchronize();
    
    // Copy hull points to device
    float *d_hull_x, *d_hull_y;
    cudaMalloc(&d_hull_x, M * sizeof(float));
    cudaMalloc(&d_hull_y, M * sizeof(float));
    cudaMemcpy(d_hull_x, hull_x, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hull_y, hull_y, M * sizeof(float), cudaMemcpyHostToDevice);
    
    // Draw hull lines in red
    blocks = (M + threads - 1) / threads;
    drawHullLines<<<blocks, threads>>>(d_image, d_hull_x, d_hull_y, M,
                                        width, height, 255, 0, 0);
    cudaDeviceSynchronize();
    
    // Draw hull points in blue (larger/more visible)
    drawPoints<<<blocks, threads>>>(d_image, d_hull_x, d_hull_y, M,
                                     width, height, 0, 0, 255);
    cudaDeviceSynchronize();
    
    // Copy image back to host
    unsigned char *h_image = (unsigned char*)malloc(image_size);
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    
    // Write PPM file
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        fprintf(fp, "P6\n%d %d\n255\n", width, height);
        fwrite(h_image, 1, image_size, fp);
        fclose(fp);
        printf("Visualization saved to %s\n", filename);
    } else {
        printf("Error: Could not open file %s for writing\n", filename);
    }
    
    // Cleanup
    free(h_image);
    cudaFree(d_image);
    cudaFree(d_points_x);
    cudaFree(d_points_y);
    cudaFree(d_hull_x);
    cudaFree(d_hull_y);
}
