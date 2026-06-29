cmake_minimum_required(VERSION 3.18)
project(trellis_cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- vendored, prebuilt ggml (built once in thirdparty/ggml/build) ---------
set(GGML_DIR   ${CMAKE_SOURCE_DIR}/thirdparty/ggml)
# Point at a prebuilt ggml tree; override -DGGML_BUILD=... for a Vulkan build
# (e.g. thirdparty/ggml/build-vk) without disturbing the default CUDA build.
set(GGML_BUILD ${GGML_DIR}/build CACHE PATH "prebuilt ggml build dir")
if(NOT EXISTS ${GGML_BUILD}/src/libggml.so)
  message(FATAL_ERROR "ggml not built. Run: cmake -B thirdparty/ggml/build -S thirdparty/ggml -DGGML_CUDA=ON && cmake --build thirdparty/ggml/build -j  (or -DGGML_VULKAN=ON into build-vk)")
endif()

find_library(GGML_LIB        ggml        PATHS ${GGML_BUILD}/src             NO_DEFAULT_PATH)
find_library(GGML_BASE_LIB   ggml-base   PATHS ${GGML_BUILD}/src             NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB    ggml-cpu    PATHS ${GGML_BUILD}/src             NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB   ggml-cuda   PATHS ${GGML_BUILD}/src/ggml-cuda   NO_DEFAULT_PATH)
find_library(GGML_VULKAN_LIB ggml-vulkan PATHS ${GGML_BUILD}/src/ggml-vulkan ${GGML_BUILD}/src NO_DEFAULT_PATH)

set(GGML_RPATH "${GGML_BUILD}/src;${GGML_BUILD}/src/ggml-cuda;${GGML_BUILD}/src/ggml-vulkan")

# --- CUDA language (for the custom deformable-conv kernel) ------------------
if(GGML_CUDA_LIB)
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
  src/trellis_model.cpp
  src/dit.cpp
  src/flow_runner.cpp
  src/ss_decoder.cpp
  src/mesh_glb.cpp
  src/dinov3.cpp
  src/sparse.cpp
  src/shape_decoder.cpp
  src/dual_grid.cpp
  src/preprocess.cpp
  src/uv_bake.cpp
  src/birefnet.cpp
  thirdparty/xatlas/xatlas.cpp
)
target_include_directories(trellis_core PUBLIC include ${GGML_DIR}/include ${CMAKE_SOURCE_DIR}/thirdparty/stb ${CMAKE_SOURCE_DIR}/thirdparty/xatlas)
target_link_libraries(trellis_core PUBLIC ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB})
if(GGML_CUDA_LIB)
  target_compile_definitions(trellis_core PUBLIC TRELLIS_USE_CUDA)
  target_link_libraries(trellis_core PUBLIC ${GGML_CUDA_LIB})
  target_sources(trellis_core PRIVATE src/deform_conv.cu)
  set_target_properties(trellis_core PROPERTIES CUDA_ARCHITECTURES "86;120" CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(trellis_core PUBLIC CUDA::cudart)
  message(STATUS "trellis.cpp: CUDA backend enabled (${GGML_CUDA_LIB})")
else()
  # No CUDA kernel in this build. Run BiRefNet's deformable conv on the GPU via a
  # Vulkan compute shader when ggml is using Vulkan (CPU is minutes at its larger
  # decoder scales); otherwise on the CPU, OpenMP-parallel over output channels.
  target_sources(trellis_core PRIVATE src/deform_conv_cpu.cpp)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(trellis_core PUBLIC OpenMP::OpenMP_CXX)
  endif()
  if(GGML_VULKAN_LIB)
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
  endif()
endif()
if(GGML_VULKAN_LIB)
  target_link_libraries(trellis_core PUBLIC ${GGML_VULKAN_LIB})
  message(STATUS "trellis.cpp: Vulkan backend enabled (${GGML_VULKAN_LIB})")
endif()

# --- executables -----------------------------------------------------------
add_executable(trellis-smoke src/smoke_test.cpp)
target_link_libraries(trellis-smoke PRIVATE trellis_core)
set_target_properties(trellis-smoke PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-flow src/test_ss_flow.cpp)
target_link_libraries(trellis-test-ss-flow PRIVATE trellis_core)
set_target_properties(trellis-test-ss-flow PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-sample src/test_ss_sample.cpp)
target_link_libraries(trellis-test-ss-sample PRIVATE trellis_core)
set_target_properties(trellis-test-ss-sample PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-birefnet src/test_birefnet.cpp)
target_link_libraries(trellis-test-birefnet PRIVATE trellis_core)
set_target_properties(trellis-test-birefnet PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-deform src/test_deform.cpp)
target_link_libraries(trellis-test-deform PRIVATE trellis_core)
set_target_properties(trellis-test-deform PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-dec src/test_ss_dec.cpp)
target_link_libraries(trellis-test-ss-dec PRIVATE trellis_core)
set_target_properties(trellis-test-ss-dec PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-slat-shape src/test_slat_shape.cpp)
target_link_libraries(trellis-test-slat-shape PRIVATE trellis_core)
set_target_properties(trellis-test-slat-shape PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-dinov3 src/test_dinov3.cpp)
target_link_libraries(trellis-test-dinov3 PRIVATE trellis_core)
set_target_properties(trellis-test-dinov3 PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-sparse-conv src/test_sparse_conv.cpp)
target_link_libraries(trellis-test-sparse-conv PRIVATE trellis_core)
set_target_properties(trellis-test-sparse-conv PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-c2s src/test_c2s.cpp)
target_link_libraries(trellis-test-c2s PRIVATE trellis_core)
set_target_properties(trellis-test-c2s PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-shape-dec src/test_shape_dec.cpp)
target_link_libraries(trellis-test-shape-dec PRIVATE trellis_core)
set_target_properties(trellis-test-shape-dec PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-cli src/trellis_cli_main.cpp src/trellis_cli.cpp src/trellis_args.cpp)
target_link_libraries(trellis-cli PRIVATE trellis_core)
set_target_properties(trellis-cli PROPERTIES BUILD_RPATH "${GGML_RPATH}")

# Resident HTTP server (keeps the process warm; wraps trellis_run).
find_package(Threads REQUIRED)
add_executable(trellis-server src/trellis-server.cpp src/trellis_cli.cpp src/trellis_args.cpp)
target_include_directories(trellis-server PRIVATE ${CMAKE_SOURCE_DIR}/thirdparty/cpp-httplib)
target_link_libraries(trellis-server PRIVATE trellis_core Threads::Threads)
set_target_properties(trellis-server PROPERTIES BUILD_RPATH "${GGML_RPATH}")

add_executable(trellis-test-ss-full src/test_ss_full.cpp)
target_link_libraries(trellis-test-ss-full PRIVATE trellis_core)
set_target_properties(trellis-test-ss-full PROPERTIES BUILD_RPATH "${GGML_RPATH}")
