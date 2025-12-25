#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

void grahamScan(float *p_x, float *p_y, int N, float *result_x, float *result_y, int *M);

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

    float *result_x = (float*) malloc(sizeof(float) * N);
    float *result_y = (float*) malloc(sizeof(float) * N);
    int M;

    grahamScan(px, py, N, result_x, result_y, &M);

    printf("Liczba punktów w otoczce: %d\n", M);
    for (int i = 0; i < (M < 10 ? M : 10); i++) {
        printf("(%f, %f)\n", result_x[i], result_y[i]);
    }

    free(px);
    free(py);
    free(result_x);
    free(result_y);

    return 0;
}
