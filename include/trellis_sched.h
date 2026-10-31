#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <string>

namespace trellis {

struct Model;

// Multi-backend execution.
//
// trellis historically ran every graph with ggml_backend_graph_compute() on a single
// backend, which requires that backend to support *every* op in the graph. That holds for
// CUDA and Vulkan, whose op coverage is near-complete. It does not hold for the Qualcomm
// Hexagon NPU: the HTP skel implements a subset of ops and rejects tensors larger than
// VTCM, and with no scheduler those ops are simply never executed -- the run reports
// "dsp-rsp: UNKNOWN" and continues with undefined data until the DSP queue desynchronizes.
//
// ggml_backend_sched assigns each node to a backend that actually supports it, keeping
// weight-consuming ops on the backend that owns the weights and inserting copies where a
// node has to run elsewhere. That is what makes partial-coverage accelerators usable.
//
// Mode: -1 auto (on only for an HTP primary),
//        0 off (legacy single-backend path), 1 on.
void set_sched_mode(int mode);

// Add Vulkan between an HTP primary and the CPU catch-all. Mode -1 uses the
// TRELLIS_VULKAN_FALLBACK environment fallback, 0 disables it, and 1 enables it.
void set_vulkan_fallback_mode(int mode);

bool sched_enabled_for(ggml_backend_t primary);

// One line for an error message when a graph could not be allocated: backend name, node
// count, the compute buffer the graph needs (measured, not allocated) and the device's
// free/total memory.
std::string describe_alloc_failure(ggml_backend_t backend, ggml_cgraph* graph);

// Fallback backends shared process-wide. Created on demand, never freed; they outlive the
// per-stage Models that borrow them.
ggml_backend_t fallback_cpu_backend();

// Build a scheduler for a freshly loaded model's backend, or nullptr when the single
// backend path is sufficient. Called by Model::load; the Model owns the result.
ggml_backend_sched_t make_sched(ggml_backend_t primary);

// Applies --threads / TRELLIS_THREADS to a CPU backend. Defined in trellis_model.cpp;
// declared here because the scheduler's CPU fallback needs it too -- a fallback stuck on
// ggml's 4-thread default would quietly become the bottleneck.
void tune_cpu_threads(ggml_backend_t b);

// Allocation + execution strategy for one graph.
//
// Both paths follow the same contract, which mirrors what the call sites already did:
//   GraphExec ex(m);
//   ex.alloc(graph);                       // must precede input uploads
//   ggml_backend_tensor_set(input, ...);   // graph tensors exist only after alloc
//   ex.compute(graph, "label");
class GraphExec {
public:
    explicit GraphExec(const Model& m);
    ~GraphExec();

    GraphExec(const GraphExec&)            = delete;
    GraphExec& operator=(const GraphExec&) = delete;

    // Allocate a graph. Call again only when the graph itself changes.
    bool alloc(ggml_cgraph* graph);

    ggml_status compute(ggml_cgraph* graph, const char* label);

    // Bytes of compute buffer, for the existing [stage-alloc] diagnostics.
    size_t buffer_size() const;

    bool uses_sched() const { return sched_ != nullptr; }

private:
    ggml_backend_t       backend_ = nullptr;
    ggml_gallocr_t       galloc_  = nullptr;   // single-backend path
    ggml_backend_sched_t sched_   = nullptr;   // borrowed from the Model
};

}  // namespace trellis
