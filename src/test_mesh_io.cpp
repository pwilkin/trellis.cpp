// Mesh input tests for --refine-mesh (docs/spec/31-mesh-refine.md). CPU only.
//
//   trellis-test-mesh-io <mesh.glb> [expect_V expect_F]
//
// Loads the mesh, checks V/F against the expected counts (tools/glb_metrics.py), normalises it
// (normalize_mesh_trellis) and checks the bounds, the axis swap and the inverse transform. Then
// writes small OBJ / glTF files to the temp directory: fan triangulation, negative indices, and
// malformed inputs that must fail with an error rather than crash.
#include "mesh_io.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

using namespace trellis;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++g_fail; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static std::string tmpfile(const std::string& name, const std::string& body) {
    const auto p = std::filesystem::temp_directory_path() / ("trellis-test-mesh-io-" + name);
    std::ofstream(p, std::ios::binary) << body;
    return p.string();
}

int main(int argc, char** argv) {
    if (argc > 1) {
        Mesh m; std::string err;
        const bool ok = load_mesh(argv[1], m, err);
        CHECK(ok, "load %s: %s", argv[1], err.c_str());
        if (ok) {
            printf("%s: V=%d F=%d\n", argv[1], m.V(), m.F());
            if (argc > 3) CHECK(m.V() == atoi(argv[2]) && m.F() == atoi(argv[3]),
                                "counts %d/%d, expected %s/%s", m.V(), m.F(), argv[2], argv[3]);
            float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < m.V(); ++i) for (int a = 0; a < 3; ++a) {
                lo[a] = std::fmin(lo[a], m.verts[i * 3 + a]); hi[a] = std::fmax(hi[a], m.verts[i * 3 + a]); }
            Mesh n = m; MeshNorm nm;
            CHECK(normalize_mesh_trellis(n, nm), "normalize");
            float nlo[3] = { 1e30f, 1e30f, 1e30f }, nhi[3] = { -1e30f, -1e30f, -1e30f };
            for (int i = 0; i < n.V(); ++i) for (int a = 0; a < 3; ++a) {
                nlo[a] = std::fmin(nlo[a], n.verts[i * 3 + a]); nhi[a] = std::fmax(nhi[a], n.verts[i * 3 + a]); }
            float ext = 0;
            for (int a = 0; a < 3; ++a) {
                CHECK(nlo[a] >= -0.5f && nhi[a] <= 0.5f, "axis %d out of [-0.5,0.5]: %g..%g", a, nlo[a], nhi[a]);
                ext = std::fmax(ext, nhi[a] - nlo[a]);
            }
            CHECK(std::fabs(ext - 0.99999f) < 1e-4f, "max extent %g", ext);
            // (x, y, z) -> (x, -z, y): grid y extent = input z extent, grid z = input y.
            CHECK(std::fabs((nhi[1] - nlo[1]) - (hi[2] - lo[2]) * nm.scale) < 1e-4f, "grid y != -input z");
            CHECK(std::fabs((nhi[2] - nlo[2]) - (hi[1] - lo[1]) * nm.scale) < 1e-4f, "grid z != input y");
            printf("normalised bounds: [%.4f %.4f] [%.4f %.4f] [%.4f %.4f], scale %.6f\n",
                   nlo[0], nhi[0], nlo[1], nhi[1], nlo[2], nhi[2], nm.scale);
            // Inverse: restore in grid axes, then the writers' (x, y, z) -> (x, z, -y) gives the input.
            std::vector<float> r = n.verts;
            restore_mesh_transform(r, nm);
            double md = 0;
            for (int i = 0; i < n.V(); ++i) {
                const float gx = r[i * 3], gy = r[i * 3 + 1], gz = r[i * 3 + 2];
                const float out[3] = { gx, gz, -gy };
                for (int a = 0; a < 3; ++a) md = std::fmax(md, std::fabs(out[a] - m.verts[i * 3 + a]));
            }
            const double span = std::fmax(hi[0] - lo[0], std::fmax(hi[1] - lo[1], hi[2] - lo[2]));
            printf("restore round trip: max |d| = %.3e (%.3e of the extent)\n", md, md / span);
            CHECK(md / span < 1e-5, "restore round trip %g", md);
        }
    }

    // OBJ: a quad (fan-split), negative indices, v/t/n tokens.
    {
        const std::string p = tmpfile("quad.obj",
            "# test\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1 4/1\nv 0 0 1\nf -5//1 -4//1 -1//1\n");
        Mesh m; std::string err;
        CHECK(load_mesh(p, m, err) && m.V() == 5 && m.F() == 3, "obj quad: %s V=%d F=%d", err.c_str(), m.V(), m.F());
        if (m.F() == 3) CHECK(m.faces[0] == 0 && m.faces[1] == 1 && m.faces[2] == 2 &&
                              m.faces[3] == 0 && m.faces[4] == 2 && m.faces[5] == 3 &&
                              m.faces[6] == 0 && m.faces[7] == 1 && m.faces[8] == 4, "obj fan/negative indices");
    }
    // Minimal glTF (embedded base64 buffer) with a translated node.
    {
        // 3 float3 positions (36 bytes) + 3 uint16 indices (6 bytes) + pad 2.
        const unsigned char bytes[44] = {
            0,0,0,0, 0,0,0,0, 0,0,0,0,   0,0,128,63, 0,0,0,0, 0,0,0,0,   0,0,0,0, 0,0,128,63, 0,0,0,0,
            0,0, 1,0, 2,0, 0,0 };
        static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string enc;
        for (int i = 0; i < 44; i += 3) {
            const int n = (bytes[i] << 16) | ((i + 1 < 44 ? bytes[i + 1] : 0) << 8) | (i + 2 < 44 ? bytes[i + 2] : 0);
            enc += b64[(n >> 18) & 63]; enc += b64[(n >> 12) & 63];
            enc += i + 1 < 44 ? b64[(n >> 6) & 63] : '='; enc += i + 2 < 44 ? b64[n & 63] : '=';
        }
        const std::string js =
            "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,\"scenes\":[{\"nodes\":[0]}],"
            "\"nodes\":[{\"mesh\":0,\"translation\":[10,0,0]}],"
            "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},\"indices\":1}]}],"
            "\"buffers\":[{\"byteLength\":44,\"uri\":\"data:application/octet-stream;base64," + enc + "\"}],"
            "\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36},{\"buffer\":0,\"byteOffset\":36,\"byteLength\":6}],"
            "\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\",\"min\":[0,0,0],\"max\":[1,1,0]},"
            "{\"bufferView\":1,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}]}";
        const std::string p = tmpfile("tri.gltf", js);
        Mesh m; std::string err;
        const bool ok = load_mesh(p, m, err);
        CHECK(ok && m.V() == 3 && m.F() == 1, "gltf: %s", err.c_str());
        if (ok && m.V() == 3) CHECK(m.verts[0] == 10.0f && m.verts[3] == 11.0f, "gltf node translation not applied");
    }
    // Malformed inputs: an error, never a crash.
    const std::pair<const char*, const char*> bad[] = {
        { "empty.obj", "" },
        { "noface.obj", "v 0 0 0\nv 1 0 0\n" },
        { "badidx.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n" },
        { "zeroidx.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n" },
        { "trunc.glb", "glTF\x02" },
        { "junk.glb", "glTFjunkjunkjunkjunk" },
        { "bad.gltf", "{\"asset\":{\"version\":\"2.0\"},\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":7}}]}]}" },
        { "x.stl", "solid x\n" },
    };
    for (const auto& b : bad) {
        const std::string p = tmpfile(b.first, b.second);
        Mesh m; std::string err;
        const bool ok = load_mesh(p, m, err);
        CHECK(!ok && !err.empty(), "malformed %s accepted", b.first);
        if (!ok) printf("malformed %-12s -> %s\n", b.first, err.c_str());
    }
    { Mesh m; std::string err; CHECK(!load_mesh("/nonexistent/x.glb", m, err), "missing file accepted"); }
    { Mesh m; m.verts = { 1, 1, 1, 1, 1, 1, 1, 1, 1 }; m.faces = { 0, 1, 2 }; MeshNorm nm;
      CHECK(!normalize_mesh_trellis(m, nm), "degenerate mesh normalised"); }

    printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL OK", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
