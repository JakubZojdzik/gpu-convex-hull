#include <bits/stdc++.h>
#include "utils.h"
using namespace std;

static constexpr float EPS = 1e-6f;

// Cross product (PQ x PR)
static inline float cross(const Point &p, const Point &q, const Point &r) {
    return (q.x - p.x) * (r.y - p.y)
         - (q.y - p.y) * (r.x - p.x);
}

static inline float dist2(const Point &p, const Point &q) {
    float dx = p.x - q.x;
    float dy = p.y - q.y;
    return dx * dx + dy * dy;
}

void grahamScan(float *p_x, float *p_y, int N,
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

    vector<Point> points(N);
    for (int i = 0; i < N; i++)
        points[i] = {p_x[i], p_y[i]};

    // Step 1: find pivot (lowest y, then lowest x)
    int pivot = 0;
    for (int i = 1; i < N; i++) {
        if (points[i].y < points[pivot].y ||
           (fabs(points[i].y - points[pivot].y) < EPS &&
            points[i].x < points[pivot].x))
        {
            pivot = i;
        }
    }

    swap(points[0], points[pivot]);
    Point p0 = points[0];

    // Step 2: sort by polar angle around p0
    sort(points.begin() + 1, points.end(),
         [&](const Point &a, const Point &b) {
             float c = cross(p0, a, b);
             if (fabs(c) < EPS)
                 return dist2(p0, a) < dist2(p0, b); // nearer first
             return c > 0; // CCW order
         });

    // Step 3: remove points collinear with p0 (keep farthest)
    vector<Point> filtered;
    filtered.reserve(N);
    filtered.push_back(p0);

    for (int i = 1; i < N; i++) {
        while (i + 1 < N &&
               fabs(cross(p0, points[i], points[i + 1])) < EPS)
        {
            i++;
        }
        filtered.push_back(points[i]);
    }

    if (filtered.size() < 3) {
        *M = (int)filtered.size();
        for (int i = 0; i < *M; i++) {
            result_x[i] = filtered[i].x;
            result_y[i] = filtered[i].y;
        }
        return;
    }

    // Step 4: Graham scan stack
    vector<Point> hull;
    hull.reserve(filtered.size());

    hull.push_back(filtered[0]);
    hull.push_back(filtered[1]);

    for (size_t i = 2; i < filtered.size(); i++) {
        while (hull.size() >= 2 &&
               cross(hull[hull.size() - 2],
                     hull[hull.size() - 1],
                     filtered[i]) <= EPS)
        {
            hull.pop_back();
        }
        hull.push_back(filtered[i]);
    }

    // Output
    *M = (int)hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
}
