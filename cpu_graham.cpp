#include <bits/stdc++.h>
#include "utils.h"
using namespace std;

// Cross product of vectors (p1-p0) and (p2-p0)
// >0 means p2 is left of line p0->p1 (counter-clockwise)
// =0 means collinear
// <0 means p2 is right of line p0->p1 (clockwise)
static float cross(const Point &p0, const Point &p1, const Point &p2) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
}

static float dist2(const Point &p, const Point &q) {
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

    // Find bottom-most point (lowest y, then leftmost x)
    int ymin = 0;
    for (int i = 1; i < N; i++)
        if (points[i].y < points[ymin].y || (points[i].y == points[ymin].y && points[i].x < points[ymin].x))
            ymin = i;

    swap(points[0], points[ymin]);
    Point p0 = points[0];

    // Sort by polar angle using atan2 (robust, transitive comparison)
    auto less_than = [p0](const Point &p1, const Point &p2) {
        float angle1 = atan2f(p1.y - p0.y, p1.x - p0.x);
        float angle2 = atan2f(p2.y - p0.y, p2.x - p0.x);
        if (angle1 != angle2) return angle1 < angle2;
        // For collinear points, closer point comes first
        return dist2(p0, p1) < dist2(p0, p2);
    };

    sort(points.begin()+1, points.end(), less_than);

    // Build hull
    vector<Point> hull;
    hull.push_back(points[0]);

    for (int i = 1; i < N; i++) {
        // Pop while we make a right turn or go straight
        while (hull.size() >= 2 && cross(hull[hull.size()-2], hull[hull.size()-1], points[i]) <= 0)
            hull.pop_back();
        hull.push_back(points[i]);
    }

    *M = hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
}

