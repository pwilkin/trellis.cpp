include(FetchContent)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

FetchContent_Declare(trellis_lemon
  URL https://lemon.cs.elte.hu/pub/sources/lemon-1.3.1.tar.gz
  URL_HASH SHA256=71b7c725f4c0b4a8ccb92eb87b208701586cf7a96156ebd821ca3ed855bad3c8)
FetchContent_GetProperties(trellis_lemon)
if(NOT trellis_lemon_POPULATED)
  FetchContent_Populate(trellis_lemon)
endif()

set(TRELLIS_LEMON_GENERATED ${CMAKE_BINARY_DIR}/retopo-deps/lemon)
file(MAKE_DIRECTORY ${TRELLIS_LEMON_GENERATED}/lemon)
function(trellis_configure_lemon)
  set(PROJECT_VERSION 1.3.1)
  set(LEMON_HAVE_LONG_LONG 1)
  configure_file(${trellis_lemon_SOURCE_DIR}/lemon/config.h.in
    ${TRELLIS_LEMON_GENERATED}/lemon/config.h)
endfunction()
trellis_configure_lemon()
add_library(trellis_lemon STATIC
  ${trellis_lemon_SOURCE_DIR}/lemon/arg_parser.cc
  ${trellis_lemon_SOURCE_DIR}/lemon/base.cc
  ${trellis_lemon_SOURCE_DIR}/lemon/color.cc
  ${trellis_lemon_SOURCE_DIR}/lemon/lp_base.cc
  ${trellis_lemon_SOURCE_DIR}/lemon/lp_skeleton.cc
  ${trellis_lemon_SOURCE_DIR}/lemon/random.cc
  ${trellis_lemon_SOURCE_DIR}/lemon/bits/windows.cc)
target_include_directories(trellis_lemon PUBLIC
  ${trellis_lemon_SOURCE_DIR} ${TRELLIS_LEMON_GENERATED})

if(TRELLIS_RETOPO_COLLISION)
  include(ExternalProject)
  FetchContent_Declare(trellis_cgal
    URL https://github.com/CGAL/cgal/releases/download/v6.1.1/CGAL-6.1.1-library.tar.xz
    URL_HASH SHA256=37e9fffe48a83209b070e1914c6aa0a7bae8076749712ab78b53245e176e0e0e)
  FetchContent_Declare(trellis_boost
    URL https://archives.boost.io/release/1.90.0/source/boost_1_90_0.tar.gz
    URL_HASH SHA256=5e93d582aff26868d581a52ae78c7d8edf3f3064742c6e77901a1f18a437eea9)
  foreach(dependency IN ITEMS trellis_cgal trellis_boost)
    FetchContent_GetProperties(${dependency})
    if(NOT ${dependency}_POPULATED)
      FetchContent_Populate(${dependency})
    endif()
  endforeach()

  add_library(trellis_cgal_deps INTERFACE)
  target_include_directories(trellis_cgal_deps INTERFACE
    ${trellis_cgal_SOURCE_DIR}/include ${trellis_boost_SOURCE_DIR})
  if(WIN32)
    target_compile_definitions(trellis_cgal_deps INTERFACE
      CGAL_DISABLE_GMP=1 CGAL_USE_BOOST_MP=1 BOOST_ALL_NO_LIB=1)
  else()
    FetchContent_Declare(trellis_gmp_source
      URL https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz
      URL_HASH SHA256=a3c2b80201b89e68616f4ad30bc66aee4927c3ce50e33929ca819d5c43538898)
    FetchContent_Declare(trellis_mpfr_source
      URL https://www.mpfr.org/mpfr-4.2.2/mpfr-4.2.2.tar.xz
      URL_HASH SHA256=b67ba0383ef7e8a8563734e2e889ef5ec3c3b898a01d00fa0a6869ad81c6ce01)
    foreach(dependency IN ITEMS trellis_gmp_source trellis_mpfr_source)
      FetchContent_GetProperties(${dependency})
      if(NOT ${dependency}_POPULATED)
        FetchContent_Populate(${dependency})
      endif()
    endforeach()
    find_program(TRELLIS_RETOPO_MAKE NAMES gmake make REQUIRED)
    set(TRELLIS_RETOPO_PREFIX ${CMAKE_BINARY_DIR}/retopo-deps/install)
    file(MAKE_DIRECTORY ${TRELLIS_RETOPO_PREFIX}/include ${TRELLIS_RETOPO_PREFIX}/lib)
    ExternalProject_Add(trellis_gmp_build
      SOURCE_DIR ${trellis_gmp_source_SOURCE_DIR}
      BINARY_DIR ${CMAKE_BINARY_DIR}/retopo-deps/gmp-build
      DOWNLOAD_COMMAND ""
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} "CFLAGS=-O2 -std=gnu17"
        <SOURCE_DIR>/configure --prefix=${TRELLIS_RETOPO_PREFIX}
        --enable-static --disable-shared --with-pic
      BUILD_COMMAND ${TRELLIS_RETOPO_MAKE}
      INSTALL_COMMAND ${TRELLIS_RETOPO_MAKE} install
      BUILD_BYPRODUCTS ${TRELLIS_RETOPO_PREFIX}/lib/libgmp.a)
    ExternalProject_Add(trellis_mpfr_build
      SOURCE_DIR ${trellis_mpfr_source_SOURCE_DIR}
      BINARY_DIR ${CMAKE_BINARY_DIR}/retopo-deps/mpfr-build
      DOWNLOAD_COMMAND ""
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} "CFLAGS=-O2 -std=gnu17"
        <SOURCE_DIR>/configure --prefix=${TRELLIS_RETOPO_PREFIX}
        --with-gmp=${TRELLIS_RETOPO_PREFIX} --enable-static --disable-shared --with-pic
      BUILD_COMMAND ${TRELLIS_RETOPO_MAKE}
      INSTALL_COMMAND ${TRELLIS_RETOPO_MAKE} install
      BUILD_BYPRODUCTS ${TRELLIS_RETOPO_PREFIX}/lib/libmpfr.a
      DEPENDS trellis_gmp_build)

    add_library(trellis_gmp STATIC IMPORTED GLOBAL)
    set_target_properties(trellis_gmp PROPERTIES
      IMPORTED_LOCATION ${TRELLIS_RETOPO_PREFIX}/lib/libgmp.a)
    add_dependencies(trellis_gmp trellis_gmp_build)
    add_library(trellis_mpfr STATIC IMPORTED GLOBAL)
    set_target_properties(trellis_mpfr PROPERTIES
      IMPORTED_LOCATION ${TRELLIS_RETOPO_PREFIX}/lib/libmpfr.a
      INTERFACE_LINK_LIBRARIES trellis_gmp)
    add_dependencies(trellis_mpfr trellis_mpfr_build)

    target_include_directories(trellis_cgal_deps INTERFACE
      ${TRELLIS_RETOPO_PREFIX}/include)
    target_compile_definitions(trellis_cgal_deps INTERFACE CGAL_USE_GMP=1)
    target_link_libraries(trellis_cgal_deps INTERFACE trellis_mpfr)
  endif()
endif()
