#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

void grahamScan(float *p_x, float *p_y, int N, float *result_x, float *result_y, int *M);
void quickHull(float *p_x, float *p_y, int N, float *result_x, float *result_y, int *M);

int main(int argc, const char** argv)
{
    int N = 10;
    float *px = (float*) malloc(sizeof(float) * N);
    float *py = (float*) malloc(sizeof(float) * N);
    // srand(123);
    srand(time(NULL));

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

    float *result_x1 = (float*) malloc(sizeof(float) * N);
    float *result_y1 = (float*) malloc(sizeof(float) * N);
    int M1;
    float *result_x2 = (float*) malloc(sizeof(float) * N);
    float *result_y2 = (float*) malloc(sizeof(float) * N);
    int M2;

    grahamScan(px, py, N, result_x1, result_y1, &M1);
    grahamScan(px, py, N, result_x2, result_y2, &M2);

    printf("Liczba punktów w otoczce: %d\n", M1);
    for (int i = 0; i < (M1 < 10 ? M1 : 10); i++) {
        printf("(%f, %f)\n", result_x1[i], result_y1[i]);
    }

    printf("Liczba punktów w otoczce: %d\n", M2);
    for (int i = 0; i < (M2 < 10 ? M2 : 10); i++) {
        printf("(%f, %f)\n", result_x2[i], result_y2[i]);
    }

    free(px);
    free(py);
    free(result_x1);
    free(result_y1);
    free(result_x2);
    free(result_y2);

    return 0;
}
