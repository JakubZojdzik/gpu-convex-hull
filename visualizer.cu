#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Simple RGB structure
struct RGB {
    unsigned char r, g, b;
};

// GPU kernel to clear background and render input points
__global__ void renderPointsKernel(float *points_x, float *points_y, int num_points,
                                    RGB *image, int width, int height,
                                    float min_x, float max_x, float min_y, float max_y,
                                    int point_step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Clear background to white
    if (idx < width * height) {
        image[idx] = {255, 255, 255}; // White background
    }
    
    // Draw input points (sample every point_step-th point)
    if (idx * point_step < num_points) {
        int point_idx = idx * point_step;
        float x = points_x[point_idx];
        float y = points_y[point_idx];
        
        // Map to image coordinates
        int img_x = (int)((x - min_x) / (max_x - min_x) * (width - 1));
        int img_y = height - 1 - (int)((y - min_y) / (max_y - min_y) * (height - 1));
        
        if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
            int pixel_idx = img_y * width + img_x;
            image[pixel_idx] = {100, 100, 255}; // Light blue for input points
        }
    }
}

// GPU kernel to draw hull edges using Bresenham's line algorithm
__global__ void drawHullEdgesKernel(float *hull_x, float *hull_y, int hull_size,
                                     RGB *image, int width, int height,
                                     float min_x, float max_x, float min_y, float max_y) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_idx >= hull_size) return;
    
    // Get current and next hull point (wrap around for last edge)
    int next_idx = (edge_idx + 1) % hull_size;
    
    float x1 = hull_x[edge_idx];
    float y1 = hull_y[edge_idx];
    float x2 = hull_x[next_idx];
    float y2 = hull_y[next_idx];
    
    // Map to image coordinates
    int img_x1 = (int)((x1 - min_x) / (max_x - min_x) * (width - 1));
    int img_y1 = height - 1 - (int)((y1 - min_y) / (max_y - min_y) * (height - 1));
    int img_x2 = (int)((x2 - min_x) / (max_x - min_x) * (width - 1));
    int img_y2 = height - 1 - (int)((y2 - min_y) / (max_y - min_y) * (height - 1));
    
    // Bresenham's line algorithm
    int dx = abs(img_x2 - img_x1);
    int dy = abs(img_y2 - img_y1);
    int x = img_x1;
    int y = img_y1;
    int x_inc = (img_x2 > img_x1) ? 1 : -1;
    int y_inc = (img_y2 > img_y1) ? 1 : -1;
    int error = dx - dy;
    
    // Draw thick green line (3 pixels wide)
    for (int i = 0; i <= dx + dy; i++) {
        for (int thickness_y = -1; thickness_y <= 1; thickness_y++) {
            for (int thickness_x = -1; thickness_x <= 1; thickness_x++) {
                int px = x + thickness_x;
                int py = y + thickness_y;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int pixel_idx = py * width + px;
                    image[pixel_idx] = {0, 255, 0}; // Green for hull edges
                }
            }
        }
        
        if (x == img_x2 && y == img_y2) break;
        
        if (error * 2 > -dy) {
            error -= dy;
            x += x_inc;
        }
        if (error * 2 < dx) {
            error += dx;
            y += y_inc;
        }
    }
}

// GPU kernel to draw hull vertices as red circles
__global__ void drawHullVerticesKernel(float *hull_x, float *hull_y, int hull_size,
                                        RGB *image, int width, int height,
                                        float min_x, float max_x, float min_y, float max_y) {
    int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vertex_idx >= hull_size) return;
    
    float x = hull_x[vertex_idx];
    float y = hull_y[vertex_idx];
    
    // Map to image coordinates
    int img_x = (int)((x - min_x) / (max_x - min_x) * (width - 1));
    int img_y = height - 1 - (int)((y - min_y) / (max_y - min_y) * (height - 1));
    
    // Draw filled circle with radius 6
    int radius = 6;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx*dx + dy*dy <= radius*radius) {
                int px = img_x + dx;
                int py = img_y + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int pixel_idx = py * width + px;
                    image[pixel_idx] = {255, 0, 0}; // Red for hull vertices
                }
            }
        }
    }
}

// Function to find min/max bounds of all points
void findBounds(float* points_x, float* points_y, int num_points,
                float* hull_x, float* hull_y, int hull_size,
                float* min_x, float* max_x, float* min_y, float* max_y) {
    *min_x = *max_x = points_x[0];
    *min_y = *max_y = points_y[0];
    
    // Check all input points
    for (int i = 0; i < num_points; i++) {
        if (points_x[i] < *min_x) *min_x = points_x[i];
        if (points_x[i] > *max_x) *max_x = points_x[i];
        if (points_y[i] < *min_y) *min_y = points_y[i];
        if (points_y[i] > *max_y) *max_y = points_y[i];
    }
    
    // Check hull points
    for (int i = 0; i < hull_size; i++) {
        if (hull_x[i] < *min_x) *min_x = hull_x[i];
        if (hull_x[i] > *max_x) *max_x = hull_x[i];
        if (hull_y[i] < *min_y) *min_y = hull_y[i];
        if (hull_y[i] > *max_y) *max_y = hull_y[i];
    }
    
    // Add 10% padding
    float pad_x = (*max_x - *min_x) * 0.1f;
    float pad_y = (*max_y - *min_y) * 0.1f;
    *min_x -= pad_x;
    *max_x += pad_x;
    *min_y -= pad_y;
    *max_y += pad_y;
}

// Simple PPM writer
void writePPM(const char* filename, RGB* image, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error: Could not open %s for writing\n", filename);
        return;
    }
    
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(image, sizeof(RGB), width * height, f);
    fclose(f);
}

extern "C" void visualizeConvexHull(float* input_x, float* input_y, int num_points,
                                     float* hull_x, float* hull_y, int hull_size,
                                     const char* output_filename) {
    const int width = 1920;
    const int height = 1080;
    
    printf("Visualizing convex hull: %d input points, %d hull points\n", num_points, hull_size);
    
    // Find bounds
    float min_x, max_x, min_y, max_y;
    findBounds(input_x, input_y, num_points, hull_x, hull_y, hull_size,
               &min_x, &max_x, &min_y, &max_y);
    
    printf("Bounds: x[%.3f, %.3f], y[%.3f, %.3f]\n", min_x, max_x, min_y, max_y);
    
    // Calculate sampling step for input points (limit visualization to ~20k points)
    int point_step = fmax(1, num_points / 20000);
    int sampled_points = (num_points + point_step - 1) / point_step;
    
    // Allocate device memory
    float *d_input_x, *d_input_y, *d_hull_x, *d_hull_y;
    RGB *d_image;
    
    cudaMalloc(&d_input_x, num_points * sizeof(float));
    cudaMalloc(&d_input_y, num_points * sizeof(float));
    cudaMalloc(&d_hull_x, hull_size * sizeof(float));
    cudaMalloc(&d_hull_y, hull_size * sizeof(float));
    cudaMalloc(&d_image, width * height * sizeof(RGB));
    
    // Copy data to device
    cudaMemcpy(d_input_x, input_x, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_y, input_y, num_points * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hull_x, hull_x, hull_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hull_y, hull_y, hull_size * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("Drawing %d sampled input points (every %d-th point) on GPU...\n", sampled_points, point_step);
    
    // Launch kernel to render background and input points
    int block_size = 256;
    int max_elements = fmax(sampled_points, width * height);
    int grid_size = (max_elements + block_size - 1) / block_size;
    
    renderPointsKernel<<<grid_size, block_size>>>(
        d_input_x, d_input_y, num_points,
        d_image, width, height,
        min_x, max_x, min_y, max_y, point_step);
    cudaDeviceSynchronize();
    
    // Draw hull edges
    printf("Drawing %d hull edges on GPU...\n", hull_size);
    int edge_grid = (hull_size + block_size - 1) / block_size;
    drawHullEdgesKernel<<<edge_grid, block_size>>>(
        d_hull_x, d_hull_y, hull_size,
        d_image, width, height,
        min_x, max_x, min_y, max_y);
    cudaDeviceSynchronize();
    
    // Draw hull vertices
    printf("Drawing %d hull vertices on GPU...\n", hull_size);
    drawHullVerticesKernel<<<edge_grid, block_size>>>(
        d_hull_x, d_hull_y, hull_size,
        d_image, width, height,
        min_x, max_x, min_y, max_y);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    RGB *h_image = (RGB*)malloc(width * height * sizeof(RGB));
    cudaMemcpy(h_image, d_image, width * height * sizeof(RGB), cudaMemcpyDeviceToHost);
    
    // Write PPM file
    char ppm_filename[256];
    snprintf(ppm_filename, sizeof(ppm_filename), "%s.ppm", output_filename);
    writePPM(ppm_filename, h_image, width, height);
    
    // Try to convert PPM to JPG using ImageMagick
    char convert_cmd[512];
    snprintf(convert_cmd, sizeof(convert_cmd), 
             "command -v convert >/dev/null 2>&1 && convert %s %s.jpg && rm %s || echo 'ImageMagick not found, saved as %s'", 
             ppm_filename, output_filename, ppm_filename, ppm_filename);
    int result = system(convert_cmd);
    
    if (result == 0) {
        printf("Visualization saved as %s.jpg\n", output_filename);
    } else {
        printf("Visualization saved as %s (install ImageMagick to convert to JPG)\n", ppm_filename);
    }
    
    // Cleanup
    cudaFree(d_input_x);
    cudaFree(d_input_y);
    cudaFree(d_hull_x);
    cudaFree(d_hull_y);
    cudaFree(d_image);
    free(h_image);
}