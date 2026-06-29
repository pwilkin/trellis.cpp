// Vulkan compute implementation of modulated deformable conv2d for BiRefNet's
// ASPPDeformable, used when trellis is built against Vulkan (no CUDA kernel). The
// CPU port (deform_conv2d_cpu) is correct but far too slow at BiRefNet's larger
// decoder scales (a 256x256/K=7 call is minutes on CPU), so this runs the same
// kernel on the GPU. Self-contained: a cached headless Vulkan compute context with
// device-local buffers fed through a host-visible staging buffer (the kernel re-reads
// inputs heavily, so they must live in fast device-local memory). Any failure falls
// back to the CPU kernel.
#include "deform_conv.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#ifndef TRELLIS_DEFORM_SPV_HEADER
#error "TRELLIS_DEFORM_SPV_HEADER must point at the glslc -mfmt=c output"
#endif

namespace trellis {

namespace {

static const uint32_t kDeformSpv[] =
#include TRELLIS_DEFORM_SPV_HEADER
;

struct PushConsts { int32_t Cin, H, W, Cout, K, pad, hasBias; };

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
    uint32_t mt_dev = UINT32_MAX;   // device-local
    uint32_t mt_host = UINT32_MAX;  // host-visible + coherent (staging)
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    Buf dbuf[6];                     // device-local: x, offset, mask, weight, bias, out
    Buf staging;                     // host-visible upload/download scratch
};

VkCtx g;
std::mutex g_mu;

#define VKCHK(x) do { if ((x) != VK_SUCCESS) return false; } while (0)

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
        // Never run on a software rasterizer (e.g. llvmpipe): it reports system RAM
        // as a huge device-local heap and would win a size-only heuristic while being
        // CPU-slow. Rank real GPUs first; tie-break by device-local heap size.
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;
        int rank = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 3
                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                 : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 1 : 0;

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
        uint32_t qi = UINT32_MAX;
        for (uint32_t i = 0; i < qn; ++i)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi = i; break; }
        if (qi == UINT32_MAX) continue;

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
        if (mt_dev == UINT32_MAX || mt_host == UINT32_MAX) continue;

        VkDeviceSize heap = 0;
        for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
            if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                heap = heap > mp.memoryHeaps[i].size ? heap : mp.memoryHeaps[i].size;
        if (rank > best_rank || (rank == best_rank && heap >= best_heap)) {
            best_rank = rank; best_heap = heap;
            g.phys = pd; g.qfam = qi; g.mt_dev = mt_dev; g.mt_host = mt_host;
            std::snprintf(best_name, sizeof(best_name), "%s", props.deviceName);
        }
    }
    if (g.phys != VK_NULL_HANDLE)
        fprintf(stderr, "[deform_vk] using %s\n", best_name);
    return g.phys != VK_NULL_HANDLE;
}

bool init_ctx() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "trellis-deform";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VKCHK(vkCreateInstance(&ici, nullptr, &g.instance));
    if (!pick_device()) return false;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g.qfam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VKCHK(vkCreateDevice(g.phys, &dci, nullptr, &g.dev));
    vkGetDeviceQueue(g.dev, g.qfam, 0, &g.queue);

    VkDescriptorSetLayoutBinding b[6];
    for (uint32_t i = 0; i < 6; ++i)
        b[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = 6; dlci.pBindings = b;
    VKCHK(vkCreateDescriptorSetLayout(g.dev, &dlci, nullptr, &g.dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &g.dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VKCHK(vkCreatePipelineLayout(g.dev, &plci, nullptr, &g.pl));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = sizeof(kDeformSpv); smci.pCode = kDeformSpv;
    VKCHK(vkCreateShaderModule(g.dev, &smci, nullptr, &g.shader));

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = g.shader; cpci.stage.pName = "main";
    cpci.layout = g.pl;
    VKCHK(vkCreateComputePipelines(g.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &g.pipe));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6};
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

bool run_vk(const float* x, int Cin, int H, int W,
            const float* offset, const float* mask,
            const float* weight, const float* bias, int Cout, int K, float* out) {
    const VkDeviceSize HW = (VkDeviceSize)H * W;
    const VkDeviceSize szX   = (VkDeviceSize)Cin * HW * 4;
    const VkDeviceSize szOff = (VkDeviceSize)2 * K * K * HW * 4;
    const VkDeviceSize szMsk = (VkDeviceSize)K * K * HW * 4;
    const VkDeviceSize szW   = (VkDeviceSize)Cout * Cin * K * K * 4;
    const VkDeviceSize szB   = (VkDeviceSize)(Cout > 0 ? Cout : 1) * 4;
    const VkDeviceSize szOut = (VkDeviceSize)Cout * HW * 4;
    const VkDeviceSize sizes[6] = {szX, szOff, szMsk, szW, szB, szOut};
    const VkDeviceSize sumIn = szX + szOff + szMsk + szW + szB;
    const VkDeviceSize stageSz = sumIn > szOut ? sumIn : szOut;

    const VkBufferUsageFlags devUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    for (int i = 0; i < 6; ++i)
        if (!ensure_buf(g.dbuf[i], sizes[i], devUsage, g.mt_dev, false)) return false;
    if (!ensure_buf(g.staging, stageSz,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, g.mt_host, true))
        return false;

    // Pack the five inputs back-to-back in the staging buffer.
    char* sp = (char*)g.staging.ptr;
    const VkDeviceSize inOff[5] = {0, szX, szX + szOff, szX + szOff + szMsk, szX + szOff + szMsk + szW};
    std::memcpy(sp + inOff[0], x, szX);
    std::memcpy(sp + inOff[1], offset, szOff);
    std::memcpy(sp + inOff[2], mask, szMsk);
    std::memcpy(sp + inOff[3], weight, szW);
    if (bias) std::memcpy(sp + inOff[4], bias, (size_t)Cout * 4);
    else std::memset(sp + inOff[4], 0, (size_t)szB);

    VkDescriptorBufferInfo dbi[6]; VkWriteDescriptorSet w[6];
    for (uint32_t i = 0; i < 6; ++i) {
        dbi[i] = {g.dbuf[i].buf, 0, VK_WHOLE_SIZE};
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = g.dset; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(g.dev, 6, w, 0, nullptr);

    PushConsts pc{Cin, H, W, Cout, K, K / 2, bias ? 1 : 0};
    const uint64_t total = (uint64_t)Cout * HW;
    const uint32_t groups = (uint32_t)((total + 255) / 256);

    const bool log_timing = std::getenv("TRELLIS_DBG_DEFORM") != nullptr;
    auto t0 = std::chrono::steady_clock::now();

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHK(vkResetCommandBuffer(g.cmd, 0));
    VKCHK(vkBeginCommandBuffer(g.cmd, &bi));
    // upload: staging -> device-local inputs
    for (int i = 0; i < 5; ++i) {
        VkBufferCopy bc{inOff[i], 0, sizes[i]};
        vkCmdCopyBuffer(g.cmd, g.staging.buf, g.dbuf[i].buf, 1, &bc);
    }
    mem_barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
    vkCmdBindDescriptorSets(g.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.dset, 0, nullptr);
    vkCmdPushConstants(g.cmd, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(g.cmd, groups, 1, 1);
    mem_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    // download: device-local out -> staging[0]
    VkBufferCopy oc{0, 0, szOut};
    vkCmdCopyBuffer(g.cmd, g.dbuf[5].buf, g.staging.buf, 1, &oc);
    VKCHK(vkEndCommandBuffer(g.cmd));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &g.cmd;
    VKCHK(vkResetFences(g.dev, 1, &g.fence));
    VKCHK(vkQueueSubmit(g.queue, 1, &si, g.fence));
    VKCHK(vkWaitForFences(g.dev, 1, &g.fence, VK_TRUE, UINT64_MAX));

    std::memcpy(out, g.staging.ptr, szOut);
    if (log_timing) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[deform_vk] Cin=%d Cout=%d H=%d W=%d K=%d -> %.1f ms\n",
                Cin, Cout, H, W, K, ms);
    }
    return true;
}

}  // namespace

void deform_conv2d_run(const float* x, int Cin, int H, int W,
                       const float* offset, const float* mask,
                       const float* weight, const float* bias, int Cout, int K,
                       float* out, int gpu) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g.tried) {
        g.tried = true;
        g.ok = init_ctx();
        if (!g.ok)
            fprintf(stderr, "[deform_vk] no usable Vulkan compute device; using CPU fallback\n");
    }
    if (g.ok && run_vk(x, Cin, H, W, offset, mask, weight, bias, Cout, K, out)) return;
    deform_conv2d_cpu(x, Cin, H, W, offset, mask, weight, bias, Cout, K, out, gpu);
}

}  // namespace trellis
