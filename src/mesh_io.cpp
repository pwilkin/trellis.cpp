#include "mesh_io.h"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace trellis {

namespace {

std::string lower_ext(const std::string& path) {
    const size_t d = path.find_last_of('.');
    if (d == std::string::npos) return "";
    std::string e = path.substr(d + 1);
    for (char& c : e) c = (char)std::tolower((unsigned char)c);
    return e;
}

bool load_gltf(const std::string& path, Mesh& out, std::string& err) {
    cgltf_options opt{};
    cgltf_data* data = nullptr;
    cgltf_result r = cgltf_parse_file(&opt, path.c_str(), &data);
    if (r != cgltf_result_success) { err = "cannot parse glTF: " + path; return false; }
    struct Guard { cgltf_data* d; ~Guard() { cgltf_free(d); } } guard{data};
    r = cgltf_load_buffers(&opt, data, path.c_str());
    if (r != cgltf_result_success) { err = "cannot load glTF buffers: " + path; return false; }
    if (cgltf_validate(data) != cgltf_result_success) { err = "invalid glTF: " + path; return false; }

    std::vector<float> verts;
    std::vector<int32_t> faces;
    for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
        const cgltf_node* node = &data->nodes[ni];
        if (!node->mesh) continue;
        float M[16];
        cgltf_node_transform_world(node, M);   // column-major
        for (cgltf_size pi = 0; pi < node->mesh->primitives_count; ++pi) {
            const cgltf_primitive* prim = &node->mesh->primitives[pi];
            if (prim->type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor* pos = nullptr;
            for (cgltf_size a = 0; a < prim->attributes_count; ++a)
                if (prim->attributes[a].type == cgltf_attribute_type_position) pos = prim->attributes[a].data;
            if (!pos || pos->count == 0) continue;
            const int64_t base = (int64_t)verts.size() / 3;
            if (base + (int64_t)pos->count > INT32_MAX) { err = "mesh too large"; return false; }
            std::vector<float> p(pos->count * 3);
            if (cgltf_accessor_unpack_floats(pos, p.data(), p.size()) != p.size()) {
                err = "cannot read POSITION accessor"; return false;
            }
            for (cgltf_size v = 0; v < pos->count; ++v) {
                const float x = p[v * 3], y = p[v * 3 + 1], z = p[v * 3 + 2];
                verts.push_back(M[0] * x + M[4] * y + M[8]  * z + M[12]);
                verts.push_back(M[1] * x + M[5] * y + M[9]  * z + M[13]);
                verts.push_back(M[2] * x + M[6] * y + M[10] * z + M[14]);
            }
            // A negative-determinant transform mirrors the mesh; flip winding to keep it outward.
            const float det = M[0] * (M[5] * M[10] - M[9] * M[6]) - M[4] * (M[1] * M[10] - M[9] * M[2])
                            + M[8] * (M[1] * M[6] - M[5] * M[2]);
            const bool flip = det < 0;
            auto push_tri = [&](cgltf_size a, cgltf_size b, cgltf_size c) -> bool {
                if (a >= pos->count || b >= pos->count || c >= pos->count) return false;
                faces.push_back((int32_t)(base + a));
                faces.push_back((int32_t)(base + (flip ? c : b)));
                faces.push_back((int32_t)(base + (flip ? b : c)));
                return true;
            };
            if (prim->indices) {
                const cgltf_size n = prim->indices->count / 3 * 3;
                for (cgltf_size i = 0; i < n; i += 3)
                    if (!push_tri(cgltf_accessor_read_index(prim->indices, i),
                                  cgltf_accessor_read_index(prim->indices, i + 1),
                                  cgltf_accessor_read_index(prim->indices, i + 2))) {
                        err = "index out of range in " + path; return false;
                    }
            } else {
                for (cgltf_size i = 0; i + 2 < pos->count; i += 3) push_tri(i, i + 1, i + 2);
            }
        }
    }
    if (faces.empty()) { err = "no triangle primitives in " + path; return false; }
    out.verts = std::move(verts);
    out.faces = std::move(faces);
    return true;
}

bool load_obj(const std::string& path, Mesh& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<float> verts;
    std::vector<int32_t> faces;
    std::string line;
    std::vector<int64_t> poly;
    size_t lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        if (line.size() < 2) continue;
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            float x, y, z;
            if (std::sscanf(line.c_str() + 2, "%f %f %f", &x, &y, &z) != 3 ||
                !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                err = "bad vertex at " + path + ":" + std::to_string(lineno); return false;
            }
            verts.push_back(x); verts.push_back(y); verts.push_back(z);
        } else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            poly.clear();
            std::istringstream is(line.substr(2));
            std::string tok;
            const int64_t nv = (int64_t)verts.size() / 3;
            while (is >> tok) {
                char* end = nullptr;
                const long long i = std::strtoll(tok.c_str(), &end, 10);   // "i", "i/t", "i//n", "i/t/n"
                if (end == tok.c_str() || i == 0) { err = "bad face at " + path + ":" + std::to_string(lineno); return false; }
                const int64_t idx = i > 0 ? i - 1 : nv + i;
                if (idx < 0 || idx >= nv) { err = "face index out of range at " + path + ":" + std::to_string(lineno); return false; }
                poly.push_back(idx);
            }
            for (size_t k = 2; k < poly.size(); ++k) {     // fan triangulation
                faces.push_back((int32_t)poly[0]);
                faces.push_back((int32_t)poly[k - 1]);
                faces.push_back((int32_t)poly[k]);
            }
        }
    }
    if (faces.empty()) { err = "no faces in " + path; return false; }
    if (verts.size() / 3 > (size_t)INT32_MAX) { err = "mesh too large"; return false; }
    out.verts = std::move(verts);
    out.faces = std::move(faces);
    return true;
}

}  // namespace

bool load_mesh(const std::string& path, Mesh& out, std::string& err) {
    char magic[4] = {0, 0, 0, 0};
    {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open " + path; return false; }
        const size_t n = std::fread(magic, 1, 4, f);
        std::fclose(f);
        if (n < 4) { err = "file too short: " + path; return false; }
    }
    const std::string ext = lower_ext(path);
    try {
        if (std::memcmp(magic, "glTF", 4) == 0 || ext == "glb" || ext == "gltf") return load_gltf(path, out, err);
        if (ext == "obj") return load_obj(path, out, err);
    } catch (const std::exception& e) {
        err = std::string("failed to load ") + path + ": " + e.what();
        return false;
    }
    err = "unsupported mesh format (expected .glb, .gltf or .obj): " + path;
    return false;
}

bool normalize_mesh_trellis(Mesh& m, MeshNorm& norm) {
    if (m.verts.size() < 9) return false;
    double lo[3] = { 1e300, 1e300, 1e300 }, hi[3] = { -1e300, -1e300, -1e300 };
    for (size_t i = 0; i < m.verts.size(); i += 3)
        for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], (double)m.verts[i + a]); hi[a] = std::max(hi[a], (double)m.verts[i + a]); }
    const double ext = std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] });
    if (!(ext > 0) || !std::isfinite(ext)) return false;
    for (int a = 0; a < 3; ++a) norm.center[a] = (float)((lo[a] + hi[a]) / 2);
    norm.scale = (float)(0.99999 / ext);
    for (size_t i = 0; i < m.verts.size(); i += 3) {
        const float x = (m.verts[i]     - norm.center[0]) * norm.scale;
        const float y = (m.verts[i + 1] - norm.center[1]) * norm.scale;
        const float z = (m.verts[i + 2] - norm.center[2]) * norm.scale;
        m.verts[i] = x; m.verts[i + 1] = -z; m.verts[i + 2] = y;
        for (int a = 0; a < 3; ++a) m.verts[i + a] = std::min(0.5f, std::max(-0.5f, m.verts[i + a]));
    }
    return true;
}

void restore_mesh_transform(std::vector<float>& v, const MeshNorm& norm) {
    // grid (x, y, z) = (gx, -gz_glTF, gy_glTF)  =>  glTF centre (cx, cy, cz) is (cx, -cz, cy) in grid axes.
    const float c[3] = { norm.center[0], -norm.center[2], norm.center[1] };
    const float inv = 1.0f / norm.scale;
    for (size_t i = 0; i + 2 < v.size(); i += 3)
        for (int a = 0; a < 3; ++a) v[i + a] = v[i + a] * inv + c[a];
}

}  // namespace trellis
