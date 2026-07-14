// Vulkan compute host driver for the CuMesh QEM edge-collapse simplifier. Mirrors the CUDA/HIP
// port (src/decimate_qem.cu) and the validated CPU algorithm (src/decimate_qem.cpp), but runs the
// four per-round COMPUTE kernels (0 qem / 1 cost / 2 propagate / 3 collapse, selected by a
// push-constant) on a headless Vulkan device. The GPU only does the per-round compute; the HOST
// builds CSR vertex->face adjacency, the unique undirected edge list + boundary flags, and does the
// stream compaction between rounds -- byte-for-byte the same as the CPU simplify_round tail.
//
// Structure follows deform_conv_vk.cpp: a cached headless compute context (instance/device/queue,
// one pipeline built from the SPIR-V header, a 12-binding storage descriptor set, one command
// buffer), device-local buffers fed through a host-visible staging buffer, and upload -> dispatch
// -> barrier -> download per round. The propagate kernel min-reduces a packed cost|id via a 64-bit
// atomicMin, so the device MUST expose shaderBufferInt64Atomics + shaderInt64; if it does not, the
// context init fails and decimate_qem() falls back to the CPU path.
#include "uv_bake.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifndef TRELLIS_DECIMATE_SPV_HEADER
#error "TRELLIS_DECIMATE_SPV_HEADER must point at the glslc -mfmt=c output"
#endif

namespace trellis {

namespace {

static const uint32_t kDecimateSpv[] =
#include TRELLIS_DECIMATE_SPV_HEADER
;

// push-constant block; field order MUST match layout(push_constant) PC in decimate_qem.comp.
struct PushConsts {
    int32_t kernel, V, F, E;
    float lam_len, lam_skinny, thresh;
};

// storage buffer bindings, in descriptor order (matches the shader's layout(binding=N)).
enum {
    B_VERTS = 0, B_FACES, B_OFF, B_V2F, B_EDGES, B_BOUNDARY,
    B_QEM, B_COST, B_VNEW, B_PROP, B_VDEAD, B_FDEAD, B_COUNT
};

struct Buf {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void*          ptr = nullptr;   // mapped (staging only)
    VkDeviceSize   cap = 0;
};

struct VkCtx {
    bool tried = false, ok = false;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    uint32_t mt_dev = UINT32_MAX;    // device-local
    uint32_t mt_host = UINT32_MAX;   // host-visible + coherent (staging)
    bool has_atomic_ext = false;     // VK_KHR_shader_atomic_int64 advertised (promoted in 1.2)
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buf dbuf[B_COUNT];               // device-local: the 12 storage buffers
    Buf staging;                     // host-visible upload/download scratch
};

VkCtx g;
std::mutex g_mu;

#define VKCHK(x) do { if ((x) != VK_SUCCESS) return false; } while (0)

// True iff `pd` has a compute queue, a device-local + a host-visible/coherent memory type, and
// 64-bit shader buffer atomics + shaderInt64 (both required by the propagate kernel). Fills the
// out-params on success.
bool device_usable(VkPhysicalDevice pd, uint32_t& qi_out, uint32_t& mt_dev_out,
                   uint32_t& mt_host_out, bool& has_ext_out) {
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    uint32_t qi = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }
    if (qi == UINT32_MAX) return false;

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t mt_dev = UINT32_MAX, mt_host = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if (mt_dev == UINT32_MAX && (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) mt_dev = i;
        if (mt_host == UINT32_MAX &&
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) mt_host = i;
    }
    if (mt_dev == UINT32_MAX || mt_host == UINT32_MAX) return false;

    // 64-bit atomics: shaderBufferInt64Atomics for atomicMin(prop[...]) and shaderInt64 for the
    // uint64_t arithmetic the shader packs cost|id with.
    VkPhysicalDeviceShaderAtomicInt64Features a64{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &a64;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!a64.shaderBufferInt64Atomics || !f2.features.shaderInt64) return false;

    // The atomics feature is promoted to Vulkan 1.2 core, but enabling the extension by name when
    // the driver advertises it keeps older loaders happy; skip it if not listed.
    bool has_ext = false;
    uint32_t en = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr) == VK_SUCCESS && en) {
        std::vector<VkExtensionProperties> exts(en);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
        for (auto& e : exts)
            if (std::strcmp(e.extensionName, VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME) == 0) { has_ext = true; break; }
    }

    qi_out = qi; mt_dev_out = mt_dev; mt_host_out = mt_host; has_ext_out = has_ext;
    return true;
}

bool pick_device() {
    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(g.instance, &n, nullptr) != VK_SUCCESS || n == 0) return false;
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g.instance, &n, devs.data());

    int best_rank = -1;
    VkDeviceSize best_heap = 0;
    char best_name[256] = {0};
    for (auto pd : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        // Never run on a software rasterizer (e.g. llvmpipe): it reports system RAM as a huge
        // device-local heap and would win a size-only heuristic while being CPU-slow.
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;

        uint32_t qi, mt_dev, mt_host; bool has_ext;
        if (!device_usable(pd, qi, mt_dev, mt_host, has_ext)) continue;

        int rank = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 3
                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 1 : 0;
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        VkDeviceSize heap = 0;
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                heap = heap > mp.memoryHeaps[i].size ? heap : mp.memoryHeaps[i].size;
        if (rank > best_rank || (rank == best_rank && heap >= best_heap)) {
            best_rank = rank; best_heap = heap;
            g.phys = pd; g.qfam = qi; g.mt_dev = mt_dev; g.mt_host = mt_host; g.has_atomic_ext = has_ext;
            std::snprintf(best_name, sizeof(best_name), "%s", props.deviceName);
        }
    }
    if (g.phys != VK_NULL_HANDLE)
        fprintf(stderr, "[decimate_vk] using %s\n", best_name);
    return g.phys != VK_NULL_HANDLE;
}

bool init_ctx() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "trellis-decimate";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VKCHK(vkCreateInstance(&ici, nullptr, &g.instance));
    if (!pick_device()) return false;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g.qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;

    // Enable the 64-bit atomics + shaderInt64 features the shader requires.
    VkPhysicalDeviceShaderAtomicInt64Features a64{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
    a64.shaderBufferInt64Atomics = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &a64;
    f2.features.shaderInt64 = VK_TRUE;

    const char* ext = VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;                                  // features via pNext -> pEnabledFeatures must be null
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    if (g.has_atomic_ext) { dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &ext; }
    VKCHK(vkCreateDevice(g.phys, &dci, nullptr, &g.dev));
    vkGetDeviceQueue(g.dev, g.qfam, 0, &g.queue);

    VkDescriptorSetLayoutBinding b[B_COUNT];
    for (uint32_t i = 0; i < B_COUNT; ++i)
        b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = B_COUNT; dlci.pBindings = b;
    VKCHK(vkCreateDescriptorSetLayout(g.dev, &dlci, nullptr, &g.dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &g.dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VKCHK(vkCreatePipelineLayout(g.dev, &plci, nullptr, &g.pl));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = sizeof(kDecimateSpv); smci.pCode = kDecimateSpv;
    VKCHK(vkCreateShaderModule(g.dev, &smci, nullptr, &g.shader));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = g.shader; cpci.stage.pName = "main";
    cpci.layout = g.pl;
    VKCHK(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &g.pipe));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, B_COUNT};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VKCHK(vkCreateDescriptorPool(g.dev, &dpci, nullptr, &g.dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = g.dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &g.dsl;
    VKCHK(vkAllocateDescriptorSets(g.dev, &dsai, &g.dset));

    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci2.queueFamilyIndex = g.qfam;
    VKCHK(vkCreateCommandPool(g.dev, &cpci2, nullptr, &g.cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = g.cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VKCHK(vkAllocateCommandBuffers(g.dev, &cbai, &g.cmd));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VKCHK(vkCreateFence(g.dev, &fci, nullptr, &g.fence));
    return true;
}

bool ensure_buf(Buf& b, VkDeviceSize bytes, VkBufferUsageFlags usage, uint32_t memtype, bool map) {
    if (bytes == 0) bytes = 4;                       // never create a zero-size buffer
    if (b.cap >= bytes && b.buf != VK_NULL_HANDLE) return true;
    if (b.ptr) { vkUnmapMemory(g.dev, b.mem); b.ptr = nullptr; }
    if (b.buf) vkDestroyBuffer(g.dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(g.dev, b.mem, nullptr);
    b.buf = VK_NULL_HANDLE; b.mem = VK_NULL_HANDLE; b.cap = 0;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKCHK(vkCreateBuffer(g.dev, &bci, nullptr, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g.dev, b.buf, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = memtype;
    VKCHK(vkAllocateMemory(g.dev, &mai, nullptr, &b.mem));
    VKCHK(vkBindBufferMemory(g.dev, b.buf, b.mem, 0));
    if (map) VKCHK(vkMapMemory(g.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    b.cap = bytes;
    return true;
}

void mem_barrier(VkAccessFlags src, VkAccessFlags dst, VkPipelineStageFlags ss, VkPipelineStageFlags ds) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = src; mb.dstAccessMask = dst;
    vkCmdPipelineBarrier(g.cmd, ss, ds, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// One simplify round: build CSR adjacency / edges / boundary on the host (byte-for-byte the CPU
// simplify_round preamble), run the four kernels on the GPU, download verts/faces/vdead/fdead,
// and compact on the host (drop dead verts + dead/degenerate faces, remap). Mutates verts/faces
// and V/F in place. Returns false on any Vulkan error (caller then aborts to the CPU path).
bool simplify_round_vk(std::vector<float>& verts, int& V, std::vector<int32_t>& faces, int& F,
                       float lam_len, float lam_skinny, float thresh) {
    // --- vertex -> incident face adjacency (CSR), exactly as decimate_qem.cpp ---
    std::vector<int32_t> off(V + 1, 0);
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) off[faces[3*f+k] + 1]++;
    for (int i = 0; i < V; ++i) off[i+1] += off[i];
    std::vector<int32_t> v2f(off[V]);
    { std::vector<int32_t> cur(off.begin(), off.end() - 1);
      for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) { int v = faces[3*f+k]; v2f[cur[v]++] = f; } }

    // --- unique undirected edges (e0<e1) + boundary vertices (edge used by a single face) ---
    std::unordered_map<uint64_t,int> ecount;
    ecount.reserve((size_t)F * 2);
    auto ekey = [](int a, int b) -> uint64_t { if (a > b) { int t=a; a=b; b=t; } return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b; };
    for (int f = 0; f < F; ++f) {
        int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
        ecount[ekey(a,b)]++; ecount[ekey(b,c)]++; ecount[ekey(c,a)]++;
    }
    std::vector<int32_t> boundary((size_t)V, 0);
    std::vector<int32_t> edges;                        // E*2, packed (e0,e1) with e0<e1
    edges.reserve(ecount.size() * 2);
    for (auto& kv : ecount) {
        int a = (int)(kv.first >> 32), b = (int)(kv.first & 0xffffffffu);
        if (kv.second == 1) { boundary[a] = boundary[b] = 1; }
        edges.push_back(a); edges.push_back(b);
    }
    const int E = (int)edges.size() / 2;

    // --- device buffer sizes for this round ---
    const VkDeviceSize szVerts = (VkDeviceSize)V * 3 * 4;
    const VkDeviceSize szFaces = (VkDeviceSize)F * 3 * 4;
    const VkDeviceSize szOff   = (VkDeviceSize)(V + 1) * 4;
    const VkDeviceSize szV2f   = (VkDeviceSize)off[V] * 4;
    const VkDeviceSize szEdges = (VkDeviceSize)E * 2 * 4;
    const VkDeviceSize szBnd   = (VkDeviceSize)V * 4;
    const VkDeviceSize szQem   = (VkDeviceSize)V * 10 * 4;
    const VkDeviceSize szCost  = (VkDeviceSize)E * 4;
    const VkDeviceSize szVnew  = (VkDeviceSize)E * 3 * 4;
    const VkDeviceSize szProp  = (VkDeviceSize)F * 8;      // uint64 per face
    const VkDeviceSize szVdead = (VkDeviceSize)V * 4;
    const VkDeviceSize szFdead = (VkDeviceSize)F * 4;
    const VkDeviceSize sizes[B_COUNT] = {
        szVerts, szFaces, szOff, szV2f, szEdges, szBnd,
        szQem, szCost, szVnew, szProp, szVdead, szFdead };

    const VkBufferUsageFlags devUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    for (int i = 0; i < B_COUNT; ++i)
        if (!ensure_buf(g.dbuf[i], sizes[i], devUsage, g.mt_dev, false)) return false;

    // Staging: the 6 uploads packed first, then a separate region for the 4 downloads (no aliasing).
    const VkDeviceSize upVerts = 0;
    const VkDeviceSize upFaces = upVerts + szVerts;
    const VkDeviceSize upOff   = upFaces + szFaces;
    const VkDeviceSize upV2f   = upOff   + szOff;
    const VkDeviceSize upEdges = upV2f   + szV2f;
    const VkDeviceSize upBnd   = upEdges + szEdges;
    const VkDeviceSize upTotal = upBnd   + szBnd;
    const VkDeviceSize dlVerts = upTotal;
    const VkDeviceSize dlFaces = dlVerts + szVerts;
    const VkDeviceSize dlVdead = dlFaces + szFaces;
    const VkDeviceSize dlFdead = dlVdead + szVdead;
    const VkDeviceSize stageSz = dlFdead + szFdead;
    if (!ensure_buf(g.staging, stageSz,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, g.mt_host, true))
        return false;

    char* sp = (char*)g.staging.ptr;
    std::memcpy(sp + upVerts, verts.data(),    (size_t)szVerts);
    std::memcpy(sp + upFaces, faces.data(),    (size_t)szFaces);
    std::memcpy(sp + upOff,   off.data(),      (size_t)szOff);
    if (szV2f)   std::memcpy(sp + upV2f,   v2f.data(),      (size_t)szV2f);
    if (szEdges) std::memcpy(sp + upEdges, edges.data(),    (size_t)szEdges);
    std::memcpy(sp + upBnd,   boundary.data(), (size_t)szBnd);

    // --- descriptor set -> the 12 device buffers ---
    VkDescriptorBufferInfo dbi[B_COUNT]; VkWriteDescriptorSet w[B_COUNT];
    for (uint32_t i = 0; i < B_COUNT; ++i) {
        dbi[i] = {g.dbuf[i].buf, 0, VK_WHOLE_SIZE};
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = g.dset; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(g.dev, B_COUNT, w, 0, nullptr);

    const uint32_t gV = (uint32_t)((V + 255) / 256);
    const uint32_t gE = (uint32_t)((E + 255) / 256);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHK(vkResetCommandBuffer(g.cmd, 0));
    VKCHK(vkBeginCommandBuffer(g.cmd, &bi));

    // upload: staging -> device (verts, faces, off, v2f, edges, boundary)
    struct { int idx; VkDeviceSize src, sz; } ups[6] = {
        {B_VERTS, upVerts, szVerts}, {B_FACES, upFaces, szFaces}, {B_OFF, upOff, szOff},
        {B_V2F, upV2f, szV2f}, {B_EDGES, upEdges, szEdges}, {B_BOUNDARY, upBnd, szBnd} };
    for (auto& u : ups) if (u.sz) {
        VkBufferCopy bc{u.src, 0, u.sz};
        vkCmdCopyBuffer(g.cmd, g.staging.buf, g.dbuf[u.idx].buf, 1, &bc);
    }
    // zero the per-round scratch; prop must start at UINT64_MAX (0xff.. fill) for the atomicMin.
    if (szQem)   vkCmdFillBuffer(g.cmd, g.dbuf[B_QEM].buf,   0, szQem,   0);
    if (szCost)  vkCmdFillBuffer(g.cmd, g.dbuf[B_COST].buf,  0, szCost,  0);
    if (szVnew)  vkCmdFillBuffer(g.cmd, g.dbuf[B_VNEW].buf,  0, szVnew,  0);
    if (szVdead) vkCmdFillBuffer(g.cmd, g.dbuf[B_VDEAD].buf, 0, szVdead, 0);
    if (szFdead) vkCmdFillBuffer(g.cmd, g.dbuf[B_FDEAD].buf, 0, szFdead, 0);
    if (szProp)  vkCmdFillBuffer(g.cmd, g.dbuf[B_PROP].buf,  0, szProp,  0xFFFFFFFFu);

    mem_barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, nullptr);

    PushConsts pc{0, V, F, E, lam_len, lam_skinny, thresh};
    // kernel 0: QEM per vertex
    pc.kernel = 0; vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (gV) vkCmdDispatch(g.cmd, gV, 1, 1);
    mem_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    // kernel 1: cost per edge
    pc.kernel = 1; vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (gE) vkCmdDispatch(g.cmd, gE, 1, 1);
    mem_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    // kernel 2: propagate per edge (64-bit atomicMin)
    pc.kernel = 2; vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (gE) vkCmdDispatch(g.cmd, gE, 1, 1);
    mem_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    // kernel 3: collapse per edge (mutates verts + faces in place)
    pc.kernel = 3; vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (gE) vkCmdDispatch(g.cmd, gE, 1, 1);
    mem_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // download: device -> staging (verts, faces, vdead, fdead)
    { VkBufferCopy bc{0, dlVerts, szVerts}; vkCmdCopyBuffer(g.cmd, g.dbuf[B_VERTS].buf, g.staging.buf, 1, &bc); }
    { VkBufferCopy bc{0, dlFaces, szFaces}; vkCmdCopyBuffer(g.cmd, g.dbuf[B_FACES].buf, g.staging.buf, 1, &bc); }
    { VkBufferCopy bc{0, dlVdead, szVdead}; vkCmdCopyBuffer(g.cmd, g.dbuf[B_VDEAD].buf, g.staging.buf, 1, &bc); }
    { VkBufferCopy bc{0, dlFdead, szFdead}; vkCmdCopyBuffer(g.cmd, g.dbuf[B_FDEAD].buf, g.staging.buf, 1, &bc); }
    VKCHK(vkEndCommandBuffer(g.cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &g.cmd;
    VKCHK(vkResetFences(g.dev, 1, &g.fence));
    VKCHK(vkQueueSubmit(g.queue, 1, &si, g.fence));
    VKCHK(vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, UINT64_MAX));

    // read back the mutated mesh + kill flags
    std::memcpy(verts.data(), sp + dlVerts, (size_t)szVerts);
    std::memcpy(faces.data(), sp + dlFaces, (size_t)szFaces);
    std::vector<int32_t> vdead((size_t)V), fdead((size_t)F);
    std::memcpy(vdead.data(), sp + dlVdead, (size_t)szVdead);
    std::memcpy(fdead.data(), sp + dlFdead, (size_t)szFdead);

    // --- compact vertices (drop collapsed) + faces (drop deleted/degenerate) -- CPU tail verbatim ---
    std::vector<int> vmap((size_t)V, -1); int nV = 0;
    for (int i = 0; i < V; ++i) if (!vdead[i]) vmap[i] = nV++;
    std::vector<float> nv((size_t)nV * 3);
    for (int i = 0; i < V; ++i) if (vmap[i] >= 0) { nv[3*vmap[i]] = verts[3*i]; nv[3*vmap[i]+1] = verts[3*i+1]; nv[3*vmap[i]+2] = verts[3*i+2]; }
    std::vector<int32_t> nf; nf.reserve(faces.size());
    for (int f = 0; f < F; ++f) {
        if (fdead[f]) continue;
        int a = vmap[faces[3*f]], b = vmap[faces[3*f+1]], c = vmap[faces[3*f+2]];
        if (a < 0 || b < 0 || c < 0 || a == b || b == c || a == c) continue;
        nf.push_back(a); nf.push_back(b); nf.push_back(c);
    }
    V = nV; F = (int)nf.size() / 3;
    verts.swap(nv); faces.swap(nf);
    return true;
}

}  // namespace

// C++-linkage entry point (declared in decimate_qem.cpp under TRELLIS_HAVE_VK_DECIMATE).
bool decimate_qem_vk(const std::vector<float>& in_verts, int V0,
                     const std::vector<int32_t>& in_faces, int F0,
                     int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g.tried) {
        g.tried = true;
        g.ok = init_ctx();
        if (!g.ok)
            fprintf(stderr, "[decimate_vk] no usable Vulkan compute device (needs 64-bit atomics); CPU fallback\n");
    }
    if (!g.ok) return false;
    if (V0 <= 0 || F0 <= 0) return false;
    if (F0 <= target_faces) { ov = in_verts; of = in_faces; return true; }

    std::vector<float> verts = in_verts;
    std::vector<int32_t> faces = in_faces;
    int V = V0, F = F0;

    float thresh = 1e-8f;
    const float lam_len = 1e-2f, lam_skinny = 1e-3f;
    int prevF = F, stalls = 0;
    for (int round = 0; round < 400 && F > target_faces; ++round) {
        if (!simplify_round_vk(verts, V, faces, F, lam_len, lam_skinny, thresh)) return false;
        if (F <= target_faces) break;
        int removed = prevF - F;
        if (removed <= 0) { if (++stalls >= 2) { thresh *= 10.0f; stalls = 0; } }
        else { stalls = 0; if ((float)removed / prevF < 1e-2f) thresh *= 10.0f; }
        prevF = F;
        if (thresh > 1e12f) break;   // fully collapsed within tolerance; stop escalating
    }

    if (V <= 0 || F <= 0) return false;

    // final compaction to referenced vertices only (matches decimate_qem CPU tail)
    std::vector<int> used((size_t)V, -1); int nV = 0;
    for (int f = 0; f < F; ++f)
        for (int k = 0; k < 3; ++k) { int v = faces[3*f+k]; if (v < 0 || v >= V) return false; if (used[v] < 0) used[v] = nV++; }
    ov.assign((size_t)nV * 3, 0.f);
    for (int i = 0; i < V; ++i) if (used[i] >= 0) { ov[3*used[i]] = verts[3*i]; ov[3*used[i]+1] = verts[3*i+1]; ov[3*used[i]+2] = verts[3*i+2]; }
    of.resize((size_t)F * 3);
    for (int f = 0; f < F; ++f) for (int k = 0; k < 3; ++k) of[3*f+k] = used[faces[3*f+k]];
    printf("  decimate_qem_vk(target=%d): V %d->%d, F %d->%d (thresh=%.1e)\n",
           target_faces, V0, nV, F0, F, (double)thresh);
    fflush(stdout);
    return true;
}

}  // namespace trellis
