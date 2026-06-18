// Loads a converted TRELLIS.2 GGUF, prints metadata and tensor stats.
// Proves the loader + ggml CUDA backend round-trip weights correctly.
//   trellis-smoke <model.gguf> [gpu] [tensor_name]
#include "trellis_model.h"
#include "ggml.h"

#include <cstdio>
#include <cmath>
#include <string>

static const char* type_name(ggml_tensor* t) { return ggml_type_name(t->type); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [gpu] [tensor]\n", argv[0]); return 1; }
    const std::string path = argv[1];
    const int gpu = argc > 2 ? atoi(argv[2]) : 0;

    printf("loading %s on %s\n", path.c_str(), gpu >= 0 ? "CUDA" : "CPU");
    trellis::Model m = trellis::Model::load(path, gpu);

    printf("arch        : %s\n", m.arch.c_str());
    printf("tensors     : %zu\n", m.tensors.size());
    printf("weight bytes: %.3f GB%s\n", m.total_bytes() / 1e9, m.on_gpu ? " (on GPU)" : "");
    if (!m.config_json.empty()) {
        std::string c = m.config_json;
        if (c.size() > 240) c = c.substr(0, 240) + " ...";
        printf("config      : %s\n", c.c_str());
    }

    printf("\nsample tensors:\n");
    int shown = 0;
    for (auto& [name, t] : m.tensors) {
        printf("  %-40s %-5s [", name.c_str(), type_name(t));
        for (int d = 0; d < ggml_n_dims(t); ++d) printf("%s%lld", d ? "," : "", (long long)t->ne[d]);
        printf("]\n");
        if (++shown >= 8) break;
    }

    // pick a tensor to inspect: explicit arg, else input_layer.weight, else first
    std::string pick = argc > 3 ? argv[3] : "input_layer.weight";
    ggml_tensor* t = m.try_get(pick);
    if (!t) { t = m.tensors.begin()->second; pick = m.tensors.begin()->first; }

    std::vector<float> v = trellis::tensor_to_f32(t);
    double sum = 0, sumsq = 0, amax = 0;
    for (float x : v) { sum += x; sumsq += (double)x * x; amax = std::max(amax, (double)std::fabs(x)); }
    const double mean = sum / v.size();
    const double rms  = std::sqrt(sumsq / v.size());
    printf("\ninspect '%s' (%s, n=%zu): mean=%.6f rms=%.6f absmax=%.6f\n",
           pick.c_str(), type_name(t), v.size(), mean, rms, amax);
    printf("first 6: ");
    for (size_t i = 0; i < v.size() && i < 6; ++i) printf("%.6f ", v[i]);
    printf("\n\nOK\n");

    m.free();
    return 0;
}
