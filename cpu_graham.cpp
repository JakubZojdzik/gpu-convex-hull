#include <bits/stdc++.h>
#include "utils.h"
using namespace std;

// >0 - R is on the left of PQ, =0 - R is in line with PQ, <0 - R is on the right of PQ
// Return values: 1 = left turn (counter-clockwise), 0 = collinear, -1 = right turn (clockwise)
static int orientation(const Point &p, const Point &q, const Point &r) {
    float val = (q.x - p.x)*(r.y - p.y) - (q.y - p.y)*(r.x - p.x);
    if (val == 0.0f) return 0;
    return (val > 0.0f) ? 1 : -1;
}

static float dist(const Point &p, const Point &q) {
    return (p.x - q.x)*(p.x - q.x) + (p.y - q.y)*(p.y - q.y);
}

void grahamScan(float *p_x, float *p_y, int N, float *result_x, float *result_y, int *M) {
    if (N <= 3) {
        for (int i = 0; i < N; i++) {
            result_x[i] = p_x[i];
            result_y[i] = p_y[i];
        }
        *M = N;
        return;
    }

    vector<Point> points(N);
    for (int i = 0; i < N; i++) points[i] = {p_x[i], p_y[i]};

    int ymin = 0;
    for (int i = 1; i < N; i++)
        if (points[i].y < points[ymin].y || (points[i].y == points[ymin].y && points[i].x < points[ymin].x))
            ymin = i;

    swap(points[0], points[ymin]);
    Point p0 = points[0];

    auto less_than = [p0](const Point &p1, const Point &p2) {
        int o = orientation(p0, p1, p2);
        if (o == 0) return dist(p0, p1) < dist(p0, p2);
        // place points with smaller polar angle first (counter-clockwise ordering)
        return o == 1;
    };

    sort(points.begin()+1, points.end(), less_than);

    vector<Point> hull;
    hull.push_back(points[0]);
    hull.push_back(points[1]);
    hull.push_back(points[2]);

    for (int i = 3; i < N; i++) {
        // While the sequence of last two points and the new point does not make a left turn,
        // pop the last point. We want to maintain counter-clockwise (left) turns.
        while (hull.size() >= 2 && orientation(hull[hull.size()-2], hull[hull.size()-1], points[i]) != 1)
            hull.pop_back();
        hull.push_back(points[i]);
    }

    *M = hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
}

