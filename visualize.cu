#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define MARGIN_PERCENT 0.05f
#define POINT_RADIUS 2
#define HULL_POINT_RADIUS 4

__device__ void drawCircle(unsigned char *image, int cx, int cy, int radius,
                           int width, int height, unsigned char r,
                           unsigned char g, unsigned char b) {
  for (int dy = -radius; dy <= radius; dy++) {
    for (int dx = -radius; dx <= radius; dx++) {
      if (dx * dx + dy * dy <= radius * radius) {
        int x = cx + dx;
        int y = cy + dy;

        if (x >= 0 && x < width && y >= 0 && y < height) {
          int pixel_idx = (y * width + x) * 3;
          image[pixel_idx + 0] = r;
          image[pixel_idx + 1] = g;
          image[pixel_idx + 2] = b;
        }
      }
    }
  }
}

__global__ void drawPoints(unsigned char *image, float *px, float *py, int N,
                           int width, int height, int radius, unsigned char r,
                           unsigned char g, unsigned char b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  float margin = MARGIN_PERCENT;
  float scale = 1.0f - 2.0f * margin;

  // [-1, 1] to [0, 1]
  float norm_x = (px[idx] + 1.0f) * 0.5f;
  float norm_y = (py[idx] + 1.0f) * 0.5f;

  int x = (int)((margin + norm_x * scale) * width);
  int y = (int)((margin + norm_y * scale) * height);

  drawCircle(image, x, y, radius, width, height, r, g, b);
}

__device__ void drawLineSegment(unsigned char *image, int x0, int y0, int x1,
                                int y1, int width, int height, unsigned char r,
                                unsigned char g, unsigned char b) {
  int dx = abs(x1 - x0);
  int dy = abs(y1 - y0);
  int sx = (x0 < x1) ? 1 : -1;
  int sy = (y0 < y1) ? 1 : -1;
  int err = dx - dy;

  int x = x0, y = y0;

  for (int i = 0; i < width + height; i++) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
      int pixel_idx = (y * width + x) * 3;
      image[pixel_idx + 0] = r;
      image[pixel_idx + 1] = g;
      image[pixel_idx + 2] = b;
    }

    if (x == x1 && y == y1)
      break;

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

__global__ void drawHullLines(unsigned char *image, float *hull_x,
                              float *hull_y, int M, int width, int height,
                              unsigned char r, unsigned char g,
                              unsigned char b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M)
    return;

  int next_idx = (idx + 1) % M;

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

__global__ void clearImage(unsigned char *image, int width, int height,
                           unsigned char r, unsigned char g, unsigned char b) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = width * height;
  if (idx >= total_pixels)
    return;

  int pixel_idx = idx * 3;
  image[pixel_idx + 0] = r;
  image[pixel_idx + 1] = g;
  image[pixel_idx + 2] = b;
}

extern "C" void visualizeConvexHull(float *points_x, float *points_y, int N,
                                    float *hull_x, float *hull_y, int M,
                                    const char *filename, int width,
                                    int height) {
  size_t image_size = width * height * 3 * sizeof(unsigned char);
  unsigned char *d_image;
  cudaMalloc(&d_image, image_size);

  int threads = 256;
  int blocks = (width * height + threads - 1) / threads;
  clearImage<<<blocks, threads>>>(d_image, width, height, 255, 255, 255);
  cudaDeviceSynchronize();

  float *d_points_x, *d_points_y;
  cudaMalloc(&d_points_x, N * sizeof(float));
  cudaMalloc(&d_points_y, N * sizeof(float));
  cudaMemcpy(d_points_x, points_x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_points_y, points_y, N * sizeof(float), cudaMemcpyHostToDevice);

  blocks = (N + threads - 1) / threads;
  drawPoints<<<blocks, threads>>>(d_image, d_points_x, d_points_y, N, width,
                                  height, POINT_RADIUS, 0, 0, 0);
  cudaDeviceSynchronize();

  float *d_hull_x, *d_hull_y;
  cudaMalloc(&d_hull_x, M * sizeof(float));
  cudaMalloc(&d_hull_y, M * sizeof(float));
  cudaMemcpy(d_hull_x, hull_x, M * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hull_y, hull_y, M * sizeof(float), cudaMemcpyHostToDevice);

  blocks = (M + threads - 1) / threads;
  drawHullLines<<<blocks, threads>>>(d_image, d_hull_x, d_hull_y, M, width,
                                     height, 255, 0, 0);
  cudaDeviceSynchronize();

  drawPoints<<<blocks, threads>>>(d_image, d_hull_x, d_hull_y, M, width, height,
                                  HULL_POINT_RADIUS, 0, 0, 255);
  cudaDeviceSynchronize();

  unsigned char *h_image = (unsigned char *)malloc(image_size);
  cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);

  FILE *fp = fopen(filename, "wb");
  if (fp) {
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    fwrite(h_image, 1, image_size, fp);
    fclose(fp);
    printf("Visualization saved to %s\n", filename);
  } else {
    printf("Error: Could not open file %s for writing\n", filename);
  }

  free(h_image);
  cudaFree(d_image);
  cudaFree(d_points_x);
  cudaFree(d_points_y);
  cudaFree(d_hull_x);
  cudaFree(d_hull_y);
}
