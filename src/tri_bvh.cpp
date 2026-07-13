#include "tri_bvh.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace trellis {

namespace {

inline float dot3(const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

// Ericson, Real-Time Collision Detection 5.1.5: closest point on triangle to p.
void closest_on_tri(const float* p, const float* a, const float* b, const float* c, float* out) {
    float ab[3], ac[3], ap[3];
    for (int k = 0; k < 3; ++k) { ab[k] = b[k]-a[k]; ac[k] = c[k]-a[k]; ap[k] = p[k]-a[k]; }
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) { std::memcpy(out, a, 12); return; }
    float bp[3];
    for (int k = 0; k < 3; ++k) bp[k] = p[k]-b[k];
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.f && d4 <= d3) { std::memcpy(out, b, 12); return; }
    const float vc = d1*d4 - d3*d2;
    if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
        const float v = d1 / (d1 - d3);
        for (int k = 0; k < 3; ++k) out[k] = a[k] + v*ab[k];
        return;
    }
    float cp[3];
    for (int k = 0; k < 3; ++k) cp[k] = p[k]-c[k];
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.f && d5 <= d6) { std::memcpy(out, c, 12); return; }
    const float vb = d5*d2 - d1*d6;
    if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
        const float w = d2 / (d2 - d6);
        for (int k = 0; k < 3; ++k) out[k] = a[k] + w*ac[k];
        return;
    }
    const float va = d3*d6 - d5*d4;
    if (va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        for (int k = 0; k < 3; ++k) out[k] = b[k] + w*(c[k]-b[k]);
        return;
    }
    const float denom = 1.f / (va + vb + vc);
    const float v = vb * denom, w = vc * denom;
    for (int k = 0; k < 3; ++k) out[k] = a[k] + ab[k]*v + ac[k]*w;
}

inline float box_dist2(const float* p, const float* bmin, const float* bmax) {
    float d2 = 0.f;
    for (int k = 0; k < 3; ++k) {
        const float v = p[k] < bmin[k] ? bmin[k] - p[k] : (p[k] > bmax[k] ? p[k] - bmax[k] : 0.f);
        d2 += v * v;
    }
    return d2;
}

}  // namespace

TriBvh TriBvh::build(const float* verts, int64_t V, const int32_t* faces, int64_t F) {
    (void)V;
    TriBvh t;
    t.verts_ = verts;
    t.faces_ = faces;
    if (F == 0) return t;
    t.prim_.resize((size_t)F);
    std::vector<float> cent((size_t)F * 3);
    for (int64_t f = 0; f < F; ++f) {
        t.prim_[f] = (int32_t)f;
        for (int k = 0; k < 3; ++k)
            cent[3*f+k] = (verts[3*faces[3*f]+k] + verts[3*faces[3*f+1]+k] + verts[3*faces[3*f+2]+k]) / 3.f;
    }
    t.nodes_.reserve((size_t)F * 2);

    struct Span { int32_t node, begin, end; };
    std::vector<Span> stack;
    t.nodes_.push_back({});
    stack.push_back({0, 0, (int32_t)F});
    while (!stack.empty()) {
        const Span s = stack.back(); stack.pop_back();
        Node& n0 = t.nodes_[s.node];
        float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
        for (int32_t i = s.begin; i < s.end; ++i) {
            const int32_t f = t.prim_[i];
            for (int j = 0; j < 3; ++j) {
                const float* v = &verts[3*faces[3*f+j]];
                for (int k = 0; k < 3; ++k) {
                    bmin[k] = std::min(bmin[k], v[k]);
                    bmax[k] = std::max(bmax[k], v[k]);
                }
            }
        }
        std::memcpy(n0.bmin, bmin, 12);
        std::memcpy(n0.bmax, bmax, 12);
        const int32_t cnt = s.end - s.begin;
        if (cnt <= 4) {
            n0.left = s.begin;
            n0.count = cnt;
            continue;
        }
        int axis = 0;
        float ext[3] = {bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]};
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
        const int32_t mid = s.begin + cnt / 2;
        std::nth_element(t.prim_.begin() + s.begin, t.prim_.begin() + mid, t.prim_.begin() + s.end,
                         [&cent, axis](int32_t a, int32_t b) { return cent[3*a+axis] < cent[3*b+axis]; });
        const int32_t lc = (int32_t)t.nodes_.size();
        t.nodes_[s.node].left = lc;
        t.nodes_[s.node].count = 0;
        t.nodes_.push_back({});
        t.nodes_.push_back({});
        stack.push_back({lc, s.begin, mid});
        stack.push_back({lc + 1, mid, s.end});
    }
    return t;
}

TriBvh::Hit TriBvh::closest(const float p[3], float max_dist) const {
    Hit hit;
    if (nodes_.empty()) return hit;
    hit.dist2 = max_dist * max_dist;
    struct Entry { float d2; int32_t node; };
    Entry stack[64];
    int sp = 0;
    stack[sp++] = {box_dist2(p, nodes_[0].bmin, nodes_[0].bmax), 0};
    while (sp > 0) {
        const Entry e = stack[--sp];
        if (e.d2 >= hit.dist2) continue;
        const Node& n = nodes_[e.node];
        if (n.count > 0) {
            for (int32_t i = 0; i < n.count; ++i) {
                const int32_t f = prim_[n.left + i];
                float q[3];
                closest_on_tri(p, &verts_[3*faces_[3*f]], &verts_[3*faces_[3*f+1]], &verts_[3*faces_[3*f+2]], q);
                const float dx = q[0]-p[0], dy = q[1]-p[1], dz = q[2]-p[2];
                const float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < hit.dist2) {
                    hit.dist2 = d2;
                    hit.face = f;
                    std::memcpy(hit.point, q, 12);
                }
            }
            continue;
        }
        const int32_t l = n.left, r = n.left + 1;
        float dl = box_dist2(p, nodes_[l].bmin, nodes_[l].bmax);
        float dr = box_dist2(p, nodes_[r].bmin, nodes_[r].bmax);
        int32_t first = l, second = r;
        if (dr < dl) { std::swap(dl, dr); first = r; second = l; }
        if (sp + 2 <= 64) {
            if (dr < hit.dist2) stack[sp++] = {dr, second};
            if (dl < hit.dist2) stack[sp++] = {dl, first};
        }
    }
    return hit;
}

}  // namespace trellis
