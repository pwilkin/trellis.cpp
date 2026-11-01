#include "trellis_sched.h"

#include "trellis_debug.h"
#include "trellis_model.h"

#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace trellis {

namespace {

int  g_sched_mode  = -1;   // -1 auto, 0 off, 1 on
int  g_vulkan_fallback_mode = -1; // -1 environment, 0 off, 1 on

// Match the maximum DiT and sparse graph capacity.
constexpr size_t kSchedGraphSize = 65536;

std::once_flag  g_cpu_once;
ggml_backend_t  g_cpu = nullptr;

std::once_flag              g_vulkan_once;
std::vector<ggml_backend_t> g_vulkan; // Vulkan fallbacks for an HTP primary

int resolve_mode() {
    if (g_sched_mode >= 0) return g_sched_mode;
    if (const char* e = std::getenv("TRELLIS_SCHED")) return atoi(e) > 0 ? 1 : 0;
    return -1;   // still auto
}

bool vulkan_fallback_enabled() {
    if (g_vulkan_fallback_mode >= 0) return g_vulkan_fallback_mode > 0;
    const char* value = std::getenv("TRELLIS_VULKAN_FALLBACK");
    return value && std::atoi(value) > 0;
}

}  // namespace

void set_sched_mode(int mode) { g_sched_mode = mode; }
void set_vulkan_fallback_mode(int mode) { g_vulkan_fallback_mode = mode; }

ggml_backend_t fallback_cpu_backend() {
    std::call_once(g_cpu_once, [] {
        g_cpu = ggml_backend_cpu_init();
        if (g_cpu) tune_cpu_threads(g_cpu);
    });
    return g_cpu;
}

const std::vector<ggml_backend_t>& vulkan_backends() {
    std::call_once(g_vulkan_once, [] {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
            const char* reg_name = reg ? ggml_backend_reg_name(reg) : "";
            if (std::strcmp(reg_name, "Vulkan") != 0) continue;
            if (ggml_backend_t b = ggml_backend_dev_init(d, nullptr)) {
                fprintf(stderr, "[trellis] Vulkan fallback %s (%s)\n",
                        ggml_backend_name(b), ggml_backend_dev_description(d));
                g_vulkan.push_back(b);
            }
        }
    });
    return g_vulkan;
}

bool sched_enabled_for(ggml_backend_t primary) {
    const int mode = resolve_mode();
    if (mode == 0) return false;
    if (mode == 1) return true;
    // Auto-enable only for the partial-coverage HTP backend. CUDA and Vulkan retain
    // their existing direct execution path unless the user explicitly requests a scheduler.
    return primary && std::strncmp(ggml_backend_name(primary), "HTP", 3) == 0;
}

ggml_backend_sched_t make_sched(ggml_backend_t primary) {
    if (!primary) return nullptr;

    std::vector<ggml_backend_t> backends;

    if (!sched_enabled_for(primary)) return nullptr;
    backends.push_back(primary);   // owns the weights and has highest priority
    if (vulkan_fallback_enabled() &&
        std::strncmp(ggml_backend_name(primary), "HTP", 3) == 0) {
        for (ggml_backend_t b : vulkan_backends()) backends.push_back(b);
    }
    if (!ggml_backend_is_cpu(primary)) {
        if (ggml_backend_t cpu = fallback_cpu_backend()) backends.push_back(cpu);
    }

    if (backends.size() < 2) {
        fprintf(stderr, "[trellis] only one backend available; skipping scheduler\n");
        return nullptr;
    }

    ggml_backend_sched_t s = ggml_backend_sched_new(
        backends.data(), /*bufts=*/nullptr, (int) backends.size(),
        kSchedGraphSize, /*parallel=*/false, /*op_offload=*/true);

    if (!s) {
        fprintf(stderr, "[trellis] failed to create backend scheduler; using %s alone\n", ggml_backend_name(primary));
        return nullptr;
    }

    fprintf(stderr, "[trellis] scheduler:");
    for (ggml_backend_t b : backends) fprintf(stderr, " %s", ggml_backend_name(b));
    fprintf(stderr, "  (ops fall back in that order)\n");
    return s;
}

GraphExec::GraphExec(const Model& m)
    : backend_(m.backend), sched_(m.sched) {}

GraphExec::~GraphExec() {
    if (galloc_) ggml_gallocr_free(galloc_);
    // sched_ is owned by the Model, not by us.
}

// Exact compute-buffer size the graph would need on `buft`, measured without allocating
// (ggml_gallocr_reserve_n_size). After a failed reserve there is no buffer left to ask.
static size_t graph_buffer_bytes(ggml_backend_buffer_type_t buft, ggml_cgraph* graph) {
    ggml_gallocr_t g = ggml_gallocr_new(buft);
    size_t size = 0;
    ggml_gallocr_reserve_n_size(g, graph, nullptr, nullptr, &size);
    ggml_gallocr_free(g);
    return size;
}

std::string describe_alloc_failure(ggml_backend_t backend, ggml_cgraph* graph) {
    char buf[256];
    const char* name = backend ? ggml_backend_name(backend) : "?";
    const double mib = 1024.0 * 1024.0;
    const double need = backend ? graph_buffer_bytes(ggml_backend_get_default_buffer_type(backend), graph) / mib : 0.0;
    size_t free_b = 0, total_b = 0;
    if (ggml_backend_dev_t dev = backend ? ggml_backend_get_device(backend) : nullptr)
        ggml_backend_dev_memory(dev, &free_b, &total_b);
    if (total_b)
        snprintf(buf, sizeof buf, "out of memory on %s: a %d-node graph needs a %.0f MiB buffer, device has %.0f MiB free of %.0f MiB",
                 name, ggml_graph_n_nodes(graph), need, free_b / mib, total_b / mib);
    else
        snprintf(buf, sizeof buf, "out of memory on %s: a %d-node graph needs a %.0f MiB buffer",
                 name, ggml_graph_n_nodes(graph), need);
    return buf;
}

bool GraphExec::alloc(ggml_cgraph* graph) {
    if (sched_) {
        // Clears the previous graph's assignment; without it the scheduler keeps the old
        // allocation and the freshly uploaded inputs would land in stale buffers.
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, graph)) {
            fprintf(stderr, "[trellis] scheduler failed to allocate graph (%s)\n",
                    describe_alloc_failure(backend_, graph).c_str());
            return false;
        }
        return true;
    }
    if (!galloc_) galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!ggml_gallocr_alloc_graph(galloc_, graph)) {
        fprintf(stderr, "[trellis] graph allocation failed (%s)\n",
                describe_alloc_failure(backend_, graph).c_str());
        return false;
    }
    // TRELLIS_DBG_ALLOC=1: per-graph compute buffer sizes (the diagnostic that used to live
    // in birefnet.cpp's run_graph before the GraphExec refactor). Cheap, so always available.
    static const bool dbg = std::getenv("TRELLIS_DBG_ALLOC") != nullptr;
    if (dbg) {
        ggml_tensor* out = ggml_graph_n_nodes(graph) ? ggml_graph_node(graph, ggml_graph_n_nodes(graph) - 1) : nullptr;
        fprintf(stderr, "      [alloc] %-40s nodes=%-4d buffer=%7.1f MiB  %s\n",
                out ? ggml_get_name(out) : "", ggml_graph_n_nodes(graph),
                buffer_size() / (1024.0 * 1024.0), ggml_backend_name(backend_));
    }
    return true;
}

ggml_status GraphExec::compute(ggml_cgraph* graph, const char* label) {
    if (sched_) return compute_graph_sched(sched_, graph, label);
    return compute_graph(backend_, graph, label);
}

size_t GraphExec::buffer_size() const {
    if (galloc_) return ggml_gallocr_get_buffer_size(galloc_, 0);
    return 0;
}

}  // namespace trellis
