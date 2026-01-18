#include <bits/stdc++.h>
#include "utils.h"
using namespace std;

static inline float cross(const Point &O, const Point &A, const Point &B) {
    return (A.x - O.x) * (B.y - O.y)
         - (A.y - O.y) * (B.x - O.x);
}

void monotoneChain(float *p_x, float *p_y, int N,
                float *result_x, float *result_y, int *M)
{
    if (N <= 1) {
        if (N == 1) {
            result_x[0] = p_x[0];
            result_y[0] = p_y[0];
        }
        *M = N;
        return;
    }

    vector<Point> pts(N);
    for (int i = 0; i < N; i++)
        pts[i] = {p_x[i], p_y[i]};

    sort(pts.begin(), pts.end(), [](const Point &a, const Point &b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });

    vector<Point> hull;
    hull.reserve(2 * N);

    constexpr float EPS = 1e-6f;

    // Lower hull
    for (const Point &p : pts) {
        while (hull.size() >= 2 &&
               cross(hull[hull.size() - 2],
                     hull[hull.size() - 1],
                     p) <= EPS)
        {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    // Upper hull
    size_t lower_size = hull.size();
    for (int i = (int)pts.size() - 2; i >= 0; i--) {
        const Point &p = pts[i];
        while (hull.size() > lower_size &&
               cross(hull[hull.size() - 2],
                     hull[hull.size() - 1],
                     p) <= EPS)
        {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    if (!hull.empty())
        hull.pop_back();

    *M = (int)hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
}
