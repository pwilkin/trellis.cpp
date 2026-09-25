cmake_minimum_required(VERSION 3.18)
project(trellis_cpp VERSION 0.8.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Git QUIET)
set(TRELLIS_GIT_COMMIT "")
set(TRELLIS_GIT_DIRTY 0)
set(TRELLIS_APP_VERSION "v${PROJECT_VERSION}")
string(TIMESTAMP TRELLIS_BUILD_TIMESTAMP_UTC "%Y-%m-%dT%H:%M:%SZ" UTC)
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE TRELLIS_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE TRELLIS_GIT_COMMIT_RESULT)
  if(TRELLIS_GIT_COMMIT_RESULT EQUAL 0)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      OUTPUT_VARIABLE TRELLIS_GIT_STATUS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(NOT "${TRELLIS_GIT_STATUS}" STREQUAL "")
      set(TRELLIS_GIT_DIRTY 1)
    endif()
  else()
    set(TRELLIS_GIT_COMMIT "")
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" tag --sort=-v:refname
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE TRELLIS_GIT_TAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT "${TRELLIS_GIT_TAGS}" STREQUAL "")
    string(REPLACE "\n" ";" TRELLIS_GIT_TAG_LIST "${TRELLIS_GIT_TAGS}")
    list(GET TRELLIS_GIT_TAG_LIST 0 TRELLIS_LATEST_TAG)
    if(NOT "${TRELLIS_LATEST_TAG}" STREQUAL "")
      set(TRELLIS_APP_VERSION "${TRELLIS_LATEST_TAG}")
    endif()
  endif()
endif()

if(NOT TRELLIS_APP_VERSION MATCHES "^v")
  set(TRELLIS_APP_VERSION "v${TRELLIS_APP_VERSION}")
endif()

set(TRELLIS_BUILD_BACKEND "cpu")

# --- vendored ggml submodule, built in-tree --------------------------------
# Backend is selected at configure time with the upstream flags
# (-DGGML_CUDA=ON / -DGGML_VULKAN=ON); the matching ggml-<backend> target then
# exists and we link it. DiT/sparse tensor names exceed the default 64 chars,
# so raise GGML_MAX_NAME for both ggml and our own sources (the header guards
# it with #ifndef, so a compile definition is enough — no submodule patch).
set(GGML_DIR ${CMAKE_SOURCE_DIR}/thirdparty/ggml)
add_compile_definitions(GGML_MAX_NAME=128)
# Colocate ggml's shared libs with our executables so they're found via $ORIGIN
# both in the build tree and once packaged.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_subdirectory(thirdparty/ggml)
set(GGML_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# --- CUDA language (for the custom deformable-conv kernel) ------------------
if(TARGET ggml-cuda)
  set(TRELLIS_BUILD_BACKEND "cuda")
  if(EXISTS /usr/local/cuda-13.3/bin/nvcc AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.3/bin/nvcc)
  endif()
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 120)
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)   # match ggml's shared cudart (avoid 2 runtimes -> invalid device ordinal)
  find_package(CUDAToolkit)
endif()

# --- core library ----------------------------------------------------------
add_library(trellis_core STATIC
  src/trellis_debug.cpp
  src/trellis_sched.cpp
  src/trellis_model.cpp
  src/dit.cpp
  src/flow_runner.cpp
  src/ss_decoder.cpp
  src/mesh_glb.cpp
  src/dinov3.cpp
  src/pixal3d.cpp
  src/naf.cpp
  src/sparse.cpp
  src/shape_decoder.cpp
  src/dual_grid.cpp
  src/preprocess.cpp
  src/uv_bake.cpp
  src/uv_overlap_cover.cpp
  src/birefnet.cpp
  thirdparty/xatlas/xatlas.cpp
  thirdparty/meshoptimizer/simplifier.cpp
  src/tri_bvh.cpp
  src/remesh_dc.cpp
  src/decimate_qem.cpp
  src/mesh_shape.cpp
  src/quad_refinement.cpp
  src/quad_refinement_gates.cpp
  src/quad_refinement_io.cpp
  src/quad_components.cpp
  src/quad_topology.cpp
  src/quad_component_cover.cpp
)
target_include_directories(trellis_core PUBLIC include ${GGML_DIR}/include ${CMAKE_SOURCE_DIR}/thirdparty/stb ${CMAKE_SOURCE_DIR}/thirdparty/xatlas ${CMAKE_SOURCE_DIR}/thirdparty/meshoptimizer ${CMAKE_SOURCE_DIR}/thirdparty/fqms)
target_compile_definitions(trellis_core PRIVATE TRELLIS_GIT_COMMIT="${TRELLIS_GIT_COMMIT}" TRELLIS_GIT_DIRTY=${TRELLIS_GIT_DIRTY} TRELLIS_BUILD_TIMESTAMP="${TRELLIS_BUILD_TIMESTAMP_UTC}" TRELLIS_APP_VERSION="${TRELLIS_APP_VERSION}")
target_link_libraries(trellis_core PUBLIC ggml ggml-base ggml-cpu)

add_executable(trellis-refine-core tools/mesh_quad_matching/selective_refinement_core.cpp)
target_link_libraries(trellis-refine-core PRIVATE trellis_core)
add_executable(trellis-chart-quads tools/mesh_quad_matching/chart_quad_uv.cpp)
target_link_libraries(trellis-chart-quads PRIVATE trellis_core)
add_executable(trellis-repack-uv tools/mesh_quad_matching/repack_uv.cpp)
target_link_libraries(trellis-repack-uv PRIVATE trellis_core)
add_executable(trellis-audit-quad-uv tools/mesh_quad_matching/audit_quad_uv_core.cpp)
target_link_libraries(trellis-audit-quad-uv PRIVATE trellis_core)
add_executable(trellis-bake-quads tools/mesh_quad_matching/bake_current_quad_probe.cpp)
target_link_libraries(trellis-bake-quads PRIVATE trellis_core)
add_executable(trellis-audit-bake-fidelity tools/mesh_quad_matching/audit_bake_fidelity.cpp)
target_link_libraries(trellis-audit-bake-fidelity PRIVATE trellis_core)
add_executable(trellis-select-uv-overlap-cover tools/mesh_quad_matching/select_uv_overlap_cover.cpp)
target_link_libraries(trellis-select-uv-overlap-cover PRIVATE trellis_core)
add_executable(trellis-filter-quads tools/mesh_quad_matching/filter_quad_components_core.cpp)
target_link_libraries(trellis-filter-quads PRIVATE trellis_core)
add_executable(trellis-test-quad-component-cover src/test_quad_component_cover.cpp)
target_link_libraries(trellis-test-quad-component-cover PRIVATE trellis_core)
add_executable(trellis-audit-quad-topology tools/mesh_quad_matching/audit_quad_topology.cpp)
target_link_libraries(trellis-audit-quad-topology PRIVATE trellis_core)
add_executable(trellis-test-quad-topology src/test_quad_topology.cpp)
target_link_libraries(trellis-test-quad-topology PRIVATE trellis_core)

option(TRELLIS_RETOPO_MATCHING "Build the native quad-matching stage" OFF)
option(TRELLIS_RETOPO_COLLISION "Build the exact collision-safe quad matcher" OFF)
if(TRELLIS_RETOPO_COLLISION AND NOT TRELLIS_RETOPO_MATCHING)
  message(FATAL_ERROR "TRELLIS_RETOPO_COLLISION requires TRELLIS_RETOPO_MATCHING")
endif()
if(TRELLIS_RETOPO_MATCHING)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT WIN32)
    message(FATAL_ERROR "Native retopology is supported on Linux and Windows")
  endif()
  if(WIN32 AND CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR "Native retopology requires 64-bit Windows")
  endif()
  include(thirdparty/retopo-dependencies.cmake)
  add_library(trellis_quad_matching STATIC src/quad_matching.cpp)
  target_include_directories(trellis_quad_matching PUBLIC include)
  target_compile_definitions(trellis_quad_matching PUBLIC TRELLIS_HAVE_QUAD_MATCHING)
  target_link_libraries(trellis_quad_matching PUBLIC trellis_lemon)
  if(TRELLIS_RETOPO_COLLISION)
    target_sources(trellis_quad_matching PRIVATE src/quad_collision.cpp src/quad_local_repair.cpp)
    target_link_libraries(trellis_quad_matching PRIVATE trellis_cgal_deps)
    if(TARGET trellis_mpfr_build)
      add_dependencies(trellis_quad_matching trellis_mpfr_build)
    endif()
    target_compile_definitions(trellis_quad_matching PUBLIC TRELLIS_HAVE_QUAD_COLLISION)
    add_executable(trellis-repair-core tools/mesh_quad_matching/local_quad_repair_core.cpp)
    target_link_libraries(trellis-repair-core PRIVATE trellis_core)
    add_executable(trellis-audit-uv tools/mesh_quad_matching/audit_uv.cpp)
    target_link_libraries(trellis-audit-uv PRIVATE trellis_core trellis_cgal_deps)
    if(TARGET trellis_mpfr_build)
      add_dependencies(trellis-audit-uv trellis_mpfr_build)
    endif()
    add_executable(trellis-retopo-atlas tools/mesh_quad_matching/retopo_atlas_pipeline.cpp)
    target_include_directories(trellis-retopo-atlas PRIVATE include)
    target_link_libraries(trellis-retopo-atlas PRIVATE trellis_core)
  endif()
  target_link_libraries(trellis_core PUBLIC trellis_quad_matching)
  add_executable(trellis-match-core tools/mesh_quad_matching/constrained_matching_core.cpp)
  target_link_libraries(trellis-match-core PRIVATE trellis_quad_matching)
endif()

# --- WebP textures (reference-parity GLB export) ---------------------------
# Optional lossy-WebP textures via EXT_texture_webp. The exporter only advertises
# the extension when WebP encoding is actually used for the embedded textures.
option(TRELLIS_WEBP "Encode GLB textures as WebP (EXT_texture_webp)" ON)
if(TRELLIS_WEBP)
  include(FetchContent)
  set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
  set(WEBP_BUILD_LIBWEBPMUX OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  FetchContent_Declare(libwebp
    URL https://github.com/webmproject/libwebp/archive/refs/tags/v1.5.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(libwebp)
  set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})
  target_compile_definitions(trellis_core PRIVATE TRELLIS_HAVE_WEBP)
  target_link_libraries(trellis_core PUBLIC webp)
  message(STATUS "trellis.cpp: WebP texture encoding enabled")
endif()
if(TARGET ggml-cuda)
  target_compile_definitions(trellis_core PUBLIC TRELLIS_USE_CUDA)
  target_link_libraries(trellis_core PUBLIC ggml-cuda)
  target_sources(trellis_core PRIVATE src/deform_conv.cu)
  # Self-contained CUDA QEM decimation (mirrors the CPU port; uses cub for scan/sort/RLE).
  target_sources(trellis_core PRIVATE src/decimate_qem.cu)
  if(MSVC)
    # CUDA 13.3 CCCL requires MSVC's conforming preprocessor for host compilation.
    target_compile_options(trellis_core PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
  endif()
  target_compile_definitions(trellis_core PUBLIC TRELLIS_HAVE_GPU_DECIMATE)
  set_target_properties(trellis_core PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(trellis_core PUBLIC CUDA::cudart)
  message(STATUS "trellis.cpp: CUDA backend enabled (${GGML_CUDA_LIB})")
else()
  # No CUDA kernel in this build. Run BiRefNet's deformable conv on the GPU via a
  # Vulkan compute shader when ggml is using Vulkan, or a HIP kernel when ggml is
  # using HIP (both are minutes on the CPU at the larger decoder scales); otherwise
  # on the CPU, std::thread-parallel over output channels (deform_conv_cpu.cpp) --
  # no OpenMP/libomp dependency.
  # HIP compiles deform_conv.cu below (issue #20); mirror the CUDA branch and skip
  # the CPU source there so deform_conv2d_run has a single definition.
  if(NOT TARGET ggml-hip)
    target_sources(trellis_core PRIVATE src/deform_conv_cpu.cpp)
  endif()
  if(TARGET ggml-vulkan)
    set(TRELLIS_BUILD_BACKEND "vulkan")
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(DEFORM_SPV_H ${CMAKE_CURRENT_BINARY_DIR}/deform_conv_comp.h)
    add_custom_command(
      OUTPUT ${DEFORM_SPV_H}
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O -mfmt=c --target-env=vulkan1.2
              -o ${DEFORM_SPV_H} ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_conv.comp
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/deform_conv.comp
      COMMENT "glslc deform_conv.comp -> SPIR-V (C array)")
    add_custom_target(trellis_deform_spv DEPENDS ${DEFORM_SPV_H})
    target_sources(trellis_core PRIVATE src/deform_conv_vk.cpp)
    add_dependencies(trellis_core trellis_deform_spv)
    target_include_directories(trellis_core PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(trellis_core PUBLIC Vulkan::Vulkan)
    set_source_files_properties(src/deform_conv_cpu.cpp PROPERTIES
      COMPILE_DEFINITIONS TRELLIS_DEFORM_VULKAN)
    set_source_files_properties(src/deform_conv_vk.cpp PROPERTIES
      COMPILE_DEFINITIONS "TRELLIS_DEFORM_SPV_HEADER=\"deform_conv_comp.h\"")
    # Self-contained Vulkan QEM decimation: same four per-round compute kernels as the CUDA/HIP
    # port, driven from the host (adjacency/edges/compaction). The shader needs 64-bit atomics
    # (GL_EXT_shader_atomic_int64), so compile against vulkan1.2 like deform_conv.
    set(DECIMATE_SPV_H ${CMAKE_CURRENT_BINARY_DIR}/decimate_qem_comp.h)
    add_custom_command(
      OUTPUT ${DECIMATE_SPV_H}
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O -mfmt=c --target-env=vulkan1.2
              -o ${DECIMATE_SPV_H} ${CMAKE_CURRENT_SOURCE_DIR}/src/decimate_qem.comp
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/decimate_qem.comp
      COMMENT "glslc decimate_qem.comp -> SPIR-V (C array)")
    add_custom_target(trellis_decimate_spv DEPENDS ${DECIMATE_SPV_H})
    target_sources(trellis_core PRIVATE src/decimate_qem_vk.cpp)
    add_dependencies(trellis_core trellis_decimate_spv)
    # Target-level so the dispatch block in decimate_qem.cpp is active too (mirrors how the CUDA
    # path defines TRELLIS_HAVE_GPU_DECIMATE for the whole library).
    target_compile_definitions(trellis_core PRIVATE TRELLIS_HAVE_VK_DECIMATE)
    set_source_files_properties(src/decimate_qem_vk.cpp PROPERTIES
      COMPILE_DEFINITIONS "TRELLIS_DECIMATE_SPV_HEADER=\"decimate_qem_comp.h\"")
  endif()
endif()
if(TARGET ggml-vulkan)
  set(TRELLIS_BUILD_BACKEND "vulkan")
  target_link_libraries(trellis_core PUBLIC ggml-vulkan)
  message(STATUS "trellis.cpp: Vulkan backend enabled")
endif()
if(TARGET ggml-metal)
  # macOS/Apple Silicon: ggml's Metal backend (GGML_METAL defaults ON for
  # Apple builds). The generic device-enumeration path in make_backend() picks
  # the Metal GPU automatically; the two custom kernels (deform_conv,
  # decimate_qem) run their CPU fallbacks on this backend.
  set(TRELLIS_BUILD_BACKEND "metal")
  target_link_libraries(trellis_core PUBLIC ggml-metal)
  message(STATUS "trellis.cpp: Metal backend enabled")
endif()
if(TARGET ggml-hip)
  set(TRELLIS_BUILD_BACKEND "hip")
  target_link_libraries(trellis_core PUBLIC ggml-hip)
  # Compile the self-contained QEM decimation kernel as HIP (ggml enabled the HIP
  # language via its subdirectory). hipcub headers live under the ROCm include root.
  enable_language(HIP)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH})
    else()
      set(ROCM_PATH /opt/rocm)
    endif()
  endif()
  # Enabling HIP makes every target that links trellis_core (which now carries a HIP object)
  # require HIP_ARCHITECTURES. Set the default globally so all subsequently-defined targets
  # (the executables below) and trellis_core itself inherit the configured GPU target(s).
  # The GPU decimation kernel must be built for the same arch(es) as the rest of
  # the backend. Prefer the user-provided GPU_TARGETS / AMDGPU_TARGETS (exactly
  # what ggml itself compiles for) over an auto-detected CMAKE_HIP_ARCHITECTURES:
  # on some toolchains CMAKE_HIP_ARCHITECTURES is pre-set to a value that omits the
  # user's GPU, so the old `if(NOT CMAKE_HIP_ARCHITECTURES)` guard skipped the
  # GPU_TARGETS the user passed and built the kernel for the wrong arch — hence the
  # runtime "device kernel image is invalid" launch error (issue #14), after which
  # decimation silently fell back to the slow CPU path.
  if(DEFINED GPU_TARGETS AND NOT "${GPU_TARGETS}" STREQUAL "")
    set(TRELLIS_HIP_ARCH "${GPU_TARGETS}")
  elseif(DEFINED AMDGPU_TARGETS AND NOT "${AMDGPU_TARGETS}" STREQUAL "")
    set(TRELLIS_HIP_ARCH "${AMDGPU_TARGETS}")
  elseif(CMAKE_HIP_ARCHITECTURES)
    set(TRELLIS_HIP_ARCH "${CMAKE_HIP_ARCHITECTURES}")
  else()
    set(TRELLIS_HIP_ARCH "gfx1151")
  endif()
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "${TRELLIS_HIP_ARCH}")
  endif()
  # Targets created before HIP was enabled also link trellis_core's HIP objects.
  get_property(TRELLIS_EARLY_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(TRELLIS_EARLY_TARGET IN LISTS TRELLIS_EARLY_TARGETS)
    get_target_property(TRELLIS_EARLY_TYPE ${TRELLIS_EARLY_TARGET} TYPE)
    if(TRELLIS_EARLY_TYPE STREQUAL "EXECUTABLE")
      set_target_properties(${TRELLIS_EARLY_TARGET} PROPERTIES HIP_ARCHITECTURES "${TRELLIS_HIP_ARCH}")
    endif()
  endforeach()
  # Both the QEM decimation kernel and BiRefNet's deformable conv (issue #20) build
  # as HIP so ROCm users get GPU BiRefNet instead of the minutes-long CPU fallback.
  set_source_files_properties(src/decimate_qem.cu src/deform_conv.cu PROPERTIES LANGUAGE HIP)
  target_sources(trellis_core PRIVATE src/decimate_qem.cu src/deform_conv.cu)
  target_compile_definitions(trellis_core PUBLIC TRELLIS_HAVE_GPU_DECIMATE)
  target_include_directories(trellis_core PRIVATE ${ROCM_PATH}/include)
  # Force the resolved arch(es) onto the HIP objects (and everything linking
  # trellis_core), overriding any auto-detected default.
  set_target_properties(trellis_core PROPERTIES HIP_ARCHITECTURES "${TRELLIS_HIP_ARCH}")
  set_source_files_properties(src/decimate_qem.cu src/deform_conv.cu PROPERTIES HIP_ARCHITECTURES "${TRELLIS_HIP_ARCH}")
  message(STATUS "trellis.cpp: HIP/ROCm backend enabled (GPU decimation + BiRefNet deform via HIP, arch ${TRELLIS_HIP_ARCH})")
endif()

target_compile_definitions(trellis_core PRIVATE TRELLIS_BUILD_BACKEND="${TRELLIS_BUILD_BACKEND}")

# --- executables -----------------------------------------------------------
add_executable(trellis-smoke src/smoke_test.cpp)
target_link_libraries(trellis-smoke PRIVATE trellis_core)
set_target_properties(trellis-smoke PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-devices src/list_devices.cpp)
target_link_libraries(trellis-devices PRIVATE trellis_core)
set_target_properties(trellis-devices PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-decode-replay src/decode_replay.cpp)
target_link_libraries(trellis-decode-replay PRIVATE trellis_core)
set_target_properties(trellis-decode-replay PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-flow src/test_ss_flow.cpp)
target_link_libraries(trellis-test-ss-flow PRIVATE trellis_core)
set_target_properties(trellis-test-ss-flow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-shape-flow src/test_shape_flow.cpp)
target_link_libraries(trellis-test-shape-flow PRIVATE trellis_core)
set_target_properties(trellis-test-shape-flow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Manual CUDA regression for the >2 Gi half2 mask-stride overflow in
# high-token-count 1024-cascade FlashAttention. The default case needs
# about 8.1 GiB of VRAM, so keep it out of the routine test suite.
if(TARGET ggml-cuda)
  add_executable(trellis-test-fa-mask-overflow src/test_fa_mask_overflow.cpp)
  target_link_libraries(trellis-test-fa-mask-overflow PRIVATE trellis_core)
  set_target_properties(trellis-test-fa-mask-overflow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

  # Small CUDA regression for trellis's V-range protection around the tensor-core
  # FlashAttention kernels. The optional `raw` mode demonstrates the backend bug.
  add_executable(trellis-test-fa-bf16-range src/test_fa_bf16_range.cpp)
  target_link_libraries(trellis-test-fa-bf16-range PRIVATE trellis_core)
  set_target_properties(trellis-test-fa-bf16-range PROPERTIES BUILD_RPATH "${GGML_RPATH}")
endif()

add_executable(trellis-test-ss-sample src/test_ss_sample.cpp)
target_link_libraries(trellis-test-ss-sample PRIVATE trellis_core)
set_target_properties(trellis-test-ss-sample PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-birefnet src/test_birefnet.cpp)
target_link_libraries(trellis-test-birefnet PRIVATE trellis_core)
set_target_properties(trellis-test-birefnet PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-preprocess src/test_preprocess.cpp)
target_link_libraries(trellis-test-preprocess PRIVATE trellis_core)
set_target_properties(trellis-test-preprocess PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-deform src/test_deform.cpp)
target_link_libraries(trellis-test-deform PRIVATE trellis_core)
set_target_properties(trellis-test-deform PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-dec src/test_ss_dec.cpp)
target_link_libraries(trellis-test-ss-dec PRIVATE trellis_core)
set_target_properties(trellis-test-ss-dec PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-slat-shape src/test_slat_shape.cpp)
target_link_libraries(trellis-test-slat-shape PRIVATE trellis_core)
set_target_properties(trellis-test-slat-shape PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-uv-chart-density src/test_uv_chart_density.cpp)
target_include_directories(trellis-test-uv-chart-density PRIVATE include)

add_executable(trellis-test-quad-quality src/test_quad_quality.cpp)
target_include_directories(trellis-test-quad-quality PRIVATE include)

add_executable(trellis-test-uv-normal-margin src/test_uv_normal_margin.cpp src/tri_bvh.cpp)
target_include_directories(trellis-test-uv-normal-margin PRIVATE include)

add_executable(trellis-test-voxel-pbr src/test_voxel_pbr.cpp src/tri_bvh.cpp)
target_include_directories(trellis-test-voxel-pbr PRIVATE include)

add_executable(trellis-test-uv-preset src/test_uv_preset.cpp)
target_link_libraries(trellis-test-uv-preset PRIVATE trellis_core)
set_target_properties(trellis-test-uv-preset PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-uv-overlap-cover src/test_uv_overlap_cover.cpp)
target_link_libraries(trellis-test-uv-overlap-cover PRIVATE trellis_core)

add_executable(trellis-test-dinov3 src/test_dinov3.cpp)
target_link_libraries(trellis-test-dinov3 PRIVATE trellis_core)
set_target_properties(trellis-test-dinov3 PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-sparse-conv src/test_sparse_conv.cpp)
target_link_libraries(trellis-test-sparse-conv PRIVATE trellis_core)
set_target_properties(trellis-test-sparse-conv PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-bigop src/test_bigop.cpp)
target_link_libraries(trellis-test-bigop PRIVATE trellis_core)
set_target_properties(trellis-test-bigop PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-c2s src/test_c2s.cpp)
target_link_libraries(trellis-test-c2s PRIVATE trellis_core)
set_target_properties(trellis-test-c2s PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-sparse-degenerate src/test_sparse_degenerate.cpp)
target_link_libraries(trellis-test-sparse-degenerate PRIVATE trellis_core)
set_target_properties(trellis-test-sparse-degenerate PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-shape-dec src/test_shape_dec.cpp)
target_link_libraries(trellis-test-shape-dec PRIVATE trellis_core)
set_target_properties(trellis-test-shape-dec PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-pixal3d src/test_pixal3d.cpp src/trellis_args.cpp)
target_link_libraries(trellis-test-pixal3d PRIVATE trellis_core)
set_target_properties(trellis-test-pixal3d PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-cli src/trellis_cli_main.cpp src/trellis_cli.cpp src/trellis_args.cpp)
target_link_libraries(trellis-cli PRIVATE trellis_core)
set_target_properties(trellis-cli PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Resident HTTP server (keeps the process warm; wraps trellis_run).
find_package(Threads REQUIRED)
add_executable(post-replay src/post_replay.cpp)
if(TARGET trellis-retopo-atlas)
  add_dependencies(trellis-retopo-atlas post-replay trellis-audit-quad-topology
    trellis-chart-quads trellis-audit-quad-uv trellis-audit-uv
    trellis-repack-uv trellis-bake-quads)
  add_dependencies(trellis-cli trellis-retopo-atlas)
endif()
target_link_libraries(post-replay PRIVATE trellis_core Threads::Threads)
set_target_properties(post-replay PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-server src/trellis-server.cpp src/trellis_cli.cpp src/trellis_args.cpp)
target_include_directories(trellis-server PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/cpp-httplib)
target_link_libraries(trellis-server PRIVATE trellis_core Threads::Threads)
set_target_properties(trellis-server PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-full src/test_ss_full.cpp)
target_link_libraries(trellis-test-ss-full PRIVATE trellis_core)
set_target_properties(trellis-test-ss-full PROPERTIES BUILD_RPATH "${GGML_RPATH}")
