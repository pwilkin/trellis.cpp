// trellis.cpp — GGUF model container shared by every TRELLIS.2 component.
#pragma once

#include <map>
#include <string>
#include <vector>
#include <cstdint>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend;
struct ggml_backend_buffer;

namespace trellis {

// A loaded GGUF: tensor metadata (ggml_context), the weights uploaded to a
// backend buffer (CUDA or CPU), and a name -> tensor lookup. One per checkpoint.
struct Model {
    gguf_context*         gguf   = nullptr;  // KV metadata + tensor table
    ggml_context*         meta   = nullptr;  // owns the ggml_tensor structs
    ggml_backend*         backend = nullptr; // where weights live
    ggml_backend_buffer*  buffer = nullptr;  // the weight buffer
    std::map<std::string, ggml_tensor*> tensors;
    std::string arch;          // general.architecture
    std::string config_json;   // trellis.config_json (raw model config)
    bool on_gpu = false;

    // gpu < 0 -> CPU backend; otherwise CUDA device index.
    static Model load(const std::string& path, int gpu = 0);

    ggml_tensor* get(const std::string& name) const;        // throws if missing
    ggml_tensor* try_get(const std::string& name) const;    // nullptr if missing
    bool has(const std::string& name) const;

    size_t total_bytes() const;
    void free();
};

// Read a tensor's full contents back to host as float32 (handles f16/f32).
std::vector<float> tensor_to_f32(ggml_tensor* t);

} // namespace trellis
