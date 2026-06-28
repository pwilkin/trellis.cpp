// trellis-server — resident HTTP wrapper around the TRELLIS.2 image->3D pipeline.
//
//   GET  /health     -> "ok"
//   POST /generate    multipart/form-data with an "image" file part
//                      (optional text field "seed"); returns model/gltf-binary
//
// The model directory is resolved once at startup; each request runs the full
// pipeline via trellis_run() (per-stage load/free, like trellis-cli), serialized
// by a mutex. Keeping the process resident avoids re-initializing the Vulkan
// backend (shader load) on every request. The compute device is chosen by
// make_backend (Vulkan when built without CUDA / when CUDA is hidden).
#include "trellis_run.h"
#include "httplib.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace {

const char* opt(int argc, char** argv, const char* key, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

std::string read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool write_file_bytes(const std::string& path, const std::string& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(data.data(), (std::streamsize) data.size());
    return f.good();
}

}  // namespace

int main(int argc, char** argv) {
    const std::string models = opt(argc, argv, "--models", "/media/ilintar/D_SSD/models/trellis2/gguf");
    const std::string host = opt(argc, argv, "--host", "127.0.0.1");
    const int port = atoi(opt(argc, argv, "--port", "8080"));
    const int gpu = atoi(opt(argc, argv, "--gpu", "0"));  // >=0 selects a GPU device (Vulkan picks the roomiest)

    std::mutex gen_mu;
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    svr.Post("/generate", [&](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_file("image")) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'image' file part\"}", "application/json");
            return;
        }
        const auto& image = req.get_file_value("image");
        uint32_t seed = 42;
        if (req.has_file("seed")) seed = (uint32_t) atoi(req.get_file_value("seed").content.c_str());

        const std::string stem = std::string(std::tmpnam(nullptr));
        const std::string in_png = stem + ".png";
        const std::string out_glb = stem + ".glb";

        std::string glb;
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            if (!write_file_bytes(in_png, image.content)) {
                res.status = 500;
                res.set_content("{\"error\":\"failed to stage input image\"}", "application/json");
                return;
            }
            fprintf(stderr, "[trellis-server] generate: %zu-byte image, seed %u\n", image.content.size(), seed);
            int rc = trellis_run(in_png, out_glb, gpu, models, seed);
            if (rc == 0) glb = read_file_bytes(out_glb);
            std::remove(in_png.c_str());
            std::remove(out_glb.c_str());
            // trellis_run also writes a sibling .ply (debug); clean it up too.
            std::remove((stem + ".ply").c_str());
            std::remove((stem + "_base.png").c_str());
        }

        if (glb.empty()) {
            res.status = 500;
            res.set_content("{\"error\":\"3D reconstruction failed\"}", "application/json");
            return;
        }
        res.set_content(glb.data(), glb.size(), "model/gltf-binary");
    });

    fprintf(stderr, "[trellis-server] models=%s gpu=%d listening on http://%s:%d\n",
            models.c_str(), gpu, host.c_str(), port);
    if (!svr.listen(host, port)) {
        fprintf(stderr, "[trellis-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
