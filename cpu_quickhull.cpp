#include <bits/stdc++.h>
#include "utils.h"

using namespace std;

// >0 - C is on the left of AB, =0 - C is in line with AB, <0 C is on the right of AB
static float orientation(const Point &a, const Point &b, const Point &c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

static float lineDist(const Point &a, const Point &b, const Point &p) {
    return fabs(orientation(a, b, p));
}

static void quickHullRec(const vector<Point> &points, const Point &a, const Point &b, int side, vector<Point> &hull) {
    int idx = -1;
    float maxDist = 0;
    vector<Point> points_above;

    for (int i = 0; i < (int)points.size(); i++) {
        float d = orientation(a, b, points[i]);
        if (side * d > 0) { // outside of the triangle
            points_above.push_back(points[i]);
            float dist = fabs(d);
            if (dist > maxDist) {
                idx = i;
                maxDist = dist;
            }
        }
    }

    // no points outside of the triangle
    if (idx == -1) {
        hull.push_back(a);
        hull.push_back(b);
        return;
    }

    quickHullRec(points_above, points[idx], a, side, hull);
    quickHullRec(points_above, points[idx], b, side, hull);
}

void quickHull(float *p_x, float *p_y, int N, float *result_x, float *result_y, int *M) {
    if (N <= 3) {
        for (int i = 0; i < N; i++) {
            result_x[i] = p_x[i];
            result_y[i] = p_y[i];
        }
        *M = N;
        return;
    }

    vector<Point> points(N);
    for (int i = 0; i < N; i++)
        points[i] = {p_x[i], p_y[i]};

    int minX = 0, maxX = 0;
    for (int i = 1; i < N; i++) {
        if (points[i].x < points[minX].x) minX = i;
        if (points[i].x > points[maxX].x) maxX = i;
    }

    vector<Point> hull;

    quickHullRec(points, points[minX], points[maxX], 1, hull);
    quickHullRec(points, points[minX], points[maxX], -1, hull);

    sort(hull.begin(), hull.end(), [](const Point &a, const Point &b) {
        if (a.x != b.x) return a.x < b.x;
        return a.y < b.y;
    });

    hull.erase(unique(hull.begin(), hull.end(), [](const Point &a, const Point &b) {
        return a.x == b.x && a.y == b.y;
    }), hull.end());

    *M = hull.size();
    for (int i = 0; i < *M; i++) {
        result_x[i] = hull[i].x;
        result_y[i] = hull[i].y;
    }
}
