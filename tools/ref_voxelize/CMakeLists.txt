# o-voxel parity oracle (tools/ref_voxelize). Not part of the default build: enable it from the
# top level with -DTRELLIS_BUILD_REF_ORACLES=ON -DTRELLIS_OVOXEL_SRC=<TRELLIS.2>/o-voxel/src/convert
# Eigen 3.4 is fetched; the upstream source stays in its own checkout.
include(FetchContent)
FetchContent_Declare(eigen3_ref
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_GetProperties(eigen3_ref)
if(NOT eigen3_ref_POPULATED)
  FetchContent_Populate(eigen3_ref)
endif()
if(NOT TRELLIS_OVOXEL_SRC OR NOT EXISTS "${TRELLIS_OVOXEL_SRC}/flexible_dual_grid.cpp")
  message(FATAL_ERROR "TRELLIS_BUILD_REF_ORACLES needs TRELLIS_OVOXEL_SRC=<TRELLIS.2>/o-voxel/src/convert")
endif()
add_executable(trellis-ref-voxelize ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${TRELLIS_OVOXEL_SRC}/flexible_dual_grid.cpp)
target_include_directories(trellis-ref-voxelize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${TRELLIS_OVOXEL_SRC} ${eigen3_ref_SOURCE_DIR})
# Same rounding as src/mesh_voxelize.cpp: no FMA contraction (GCC fuses by default on aarch64).
target_compile_options(trellis-ref-voxelize PRIVATE -O2 $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)
