#include "trellis_model.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef TRELLIS_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace trellis {

static ggml_backend* make_backend(int gpu) {
#ifdef TRELLIS_USE_CUDA
    if (gpu >= 0) {
        ggml_backend* b = ggml_backend_cuda_init(gpu);
        if (b) return b;
        fprintf(stderr, "[trellis] CUDA init failed on device %d, falling back to CPU\n", gpu);
    }
#endif
    return ggml_backend_cpu_init();
}

Model Model::load(const std::string& path, int gpu) {
    Model m;

    ggml_context* meta = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;       // tensors are metadata only; we upload data ourselves
    gp.ctx      = &meta;
    m.gguf = gguf_init_from_file(path.c_str(), gp);
    if (!m.gguf) throw std::runtime_error("failed to open gguf: " + path);
    m.meta = meta;

    // metadata
    if (int64_t k = gguf_find_key(m.gguf, "general.architecture"); k >= 0)
        m.arch = gguf_get_val_str(m.gguf, k);
    if (int64_t k = gguf_find_key(m.gguf, "trellis.config_json"); k >= 0)
        m.config_json = gguf_get_val_str(m.gguf, k);

    m.backend = make_backend(gpu);
    m.on_gpu  = gpu >= 0;

    // allocate one buffer for every tensor declared in the file
    m.buffer = ggml_backend_alloc_ctx_tensors(meta, m.backend);
    if (!m.buffer) throw std::runtime_error("failed to allocate tensor buffer for " + path);

    // stream weights from disk into the backend buffer
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot reopen gguf: " + path);
    const size_t data_off = gguf_get_data_offset(m.gguf);
    std::vector<uint8_t> staging;
    const int64_t n = gguf_get_n_tensors(m.gguf);
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(m.gguf, i);
        ggml_tensor* t = ggml_get_tensor(meta, name);
        const size_t nbytes = ggml_nbytes(t);
        const size_t off = data_off + gguf_get_tensor_offset(m.gguf, i);
        staging.resize(nbytes);
        if (fseek(f, (long)off, SEEK_SET) != 0 ||
            fread(staging.data(), 1, nbytes, f) != nbytes) {
            fclose(f);
            throw std::runtime_error(std::string("short read for tensor ") + name);
        }
        ggml_backend_tensor_set(t, staging.data(), 0, nbytes);
        m.tensors[name] = t;
    }
    fclose(f);
    return m;
}

ggml_tensor* Model::try_get(const std::string& name) const {
    auto it = tensors.find(name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor* Model::get(const std::string& name) const {
    ggml_tensor* t = try_get(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return t;
}

bool Model::has(const std::string& name) const { return tensors.count(name) > 0; }

size_t Model::total_bytes() const {
    size_t s = 0;
    for (auto& [k, t] : tensors) s += ggml_nbytes(t);
    return s;
}

void Model::free() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (backend) ggml_backend_free(backend);
    if (gguf) gguf_free(gguf);
    if (meta) ggml_free(meta);
    buffer = nullptr; backend = nullptr; gguf = nullptr; meta = nullptr;
    tensors.clear();
}

std::vector<float> tensor_to_f32(ggml_tensor* t) {
    const int64_t ne = ggml_nelements(t);
    std::vector<float> out(ne);
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), raw.data(), ne * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t*)raw.data(), out.data(), ne);
    } else {
        throw std::runtime_error("tensor_to_f32: unsupported type");
    }
    return out;
}

} // namespace trellis
