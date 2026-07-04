cmake_minimum_required(VERSION 3.18)
project(trellis_cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
target_link_libraries(trellis_core PUBLIC ggml ggml-base ggml-cpu)
if(TARGET ggml-cuda)
  target_compile_definitions(trellis_core PUBLIC TRELLIS_USE_CUDA)
  target_link_libraries(trellis_core PUBLIC ggml-cuda)
  target_sources(trellis_core PRIVATE src/deform_conv.cu)
  set_target_properties(trellis_core PROPERTIES CUDA_ARCHITECTURES "86;120" CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(trellis_core PUBLIC CUDA::cudart)
  message(STATUS "trellis.cpp: CUDA backend enabled (${GGML_CUDA_LIB})")
else()
  # No CUDA kernel in this build. Run BiRefNet's deformable conv on the GPU via a
  # Vulkan compute shader when ggml is using Vulkan (CPU is minutes at its larger
  # decoder scales); otherwise on the CPU, std::thread-parallel over output
  # channels (see deform_conv_cpu.cpp) -- no OpenMP/libomp dependency.
  target_sources(trellis_core PRIVATE src/deform_conv_cpu.cpp)
  if(TARGET ggml-vulkan)
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
if(TARGET ggml-vulkan)
  target_link_libraries(trellis_core PUBLIC ggml-vulkan)
  message(STATUS "trellis.cpp: Vulkan backend enabled")
endif()
if(TARGET ggml-hip)
  target_link_libraries(trellis_core PUBLIC ggml-hip)
  message(STATUS "trellis.cpp: HIP/ROCm backend enabled")
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
