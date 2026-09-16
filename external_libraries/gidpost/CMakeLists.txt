cmake_minimum_required (VERSION 3.10)

if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT DEFINED CMAKE_BUILD_TYPE)

set(USE_PKGCONFIG false)
if(NOT DEFINED USE_PKGCONFIG)
  set(USE_PKGCONFIG false)
  if(DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_ROOT $ENV{VCPKG_ROOT})
    message(STATUS "VCPKG_ROOT detected: ${VCPKG_ROOT}")
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    include(${CMAKE_TOOLCHAIN_FILE})
    if(DEFINED ENV{PKG_CONFIG_PATH})
      find_package(PkgConfig REQUIRED)
      if(PKG_CONFIG_FOUND)
        set(USE_PKGCONFIG true)
        message(STATUS "    using pkg-config with PKG_CONFIG_PATH = $ENV{PKG_CONFIG_PATH}")
      endif(PKG_CONFIG_FOUND)
    else(DEFINED ENV{PKG_CONFIG_PATH})
      message(WARNING "    environment variable PKG_CONFIG_PATH not found, using cmake find_package()")
    endif(DEFINED ENV{PKG_CONFIG_PATH})
  endif(DEFINED ENV{VCPKG_ROOT})
endif(NOT DEFINED USE_PKGCONFIG)


project (gidpost)

## # Add your project files and targets here
## add_executable(MyExecutable main.cpp)
## 
## # Link against the found libraries
## target_link_libraries(MyExecutable ${ZLIB_LIBRARIES} ${HDF5_LIBRARIES})

# The gidpost version number,
# GP_VERSION_MAJOR, GP_VERSION_MINOR and GP_VERSION are defined in gidpost.h
# set (GP_VERSION_MAJOR 2)
# set (GP_VERSION_MINOR 10)
# set( GP_VERSION ${GP_VERSION_MAJOR}.${GP_VERSION_MINOR} )

# Find zlib
if(NOT USE_PKGCONFIG)
  # find_package(ZLIB REQUIRED)
  if(ZLIB_FOUND)
    set(FOUND_BY "find_package()")
  endif(ZLIB_FOUND)
endif(NOT USE_PKGCONFIG)

# Try using pkg-config
if(USE_PKGCONFIG)
  pkg_check_modules(ZLIB REQUIRED zlib)
  if(ZLIB_FOUND)
    set(FOUND_BY "pkg-config")
  endif()
endif(USE_PKGCONFIG)

if(ZLIB_FOUND)
    message(STATUS "Found ZLIB by ${FOUND_BY}: ${ZLIB_INCLUDE_DIRS}")
    include_directories(${ZLIB_INCLUDE_DIRS})
    link_directories(${ZLIB_LIBRARY_DIRS})
else(ZLIB_FOUND)
    message(FATAL_ERROR "ZLIB not found. Please install ZLIB or set the ZLIB_ROOT environment variable or make it available through pkg-config and define environment variable PKG_CONFIG_PATH.")
endif()

# Find hdf5
# find_package(HDF5 REQUIRED COMPONENTS C HL)

if( NOT DEFINED ENABLE_HDF5)
    option (ENABLE_HDF5 "Use HDF5" ON)
endif( NOT DEFINED ENABLE_HDF5)

if(ENABLE_HDF5)
  if(NOT USE_PKGCONFIG)
    find_package(HDF5 COMPONENTS C HL)
    if(HDF5_FOUND)
      set(FOUND_BY "find_package()")
    endif(HDF5_FOUND)
  endif(NOT USE_PKGCONFIG)
  
  if(USE_PKGCONFIG)
    # pkg_check separately so that we can use HDF5_LIBRARIES and HDF5_HL_LIBRARIES
    # as defined also by find_package()
    # pkg_check_modules(HDF5 REQUIRED hdf5 hdf5_hl)
    pkg_check_modules(HDF5 REQUIRED hdf5)
    if(HDF5_FOUND)
      set(FOUND_BY "pkg-config")
      set(HDF5_C_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
      ##foreach(SINGLE_LIB ${HDF5_C_INCLUDE_DIRS})
      ##  set(HDF5_C_INCLUDE_DIR ${SINGLE_LIB})
      ##  break()
      ##endforeach()
      pkg_check_modules(HDF5_HL REQUIRED hdf5 hdf5_hl)
      set(HDF5_HL_C_INCLUDE_DIRS ${HDF5_HL_INCLUDE_DIRS})
      ## foreach(SINGLE_LIB ${HDF5_HL_C_INCLUDE_DIRS})
      ##   set(HDF5_HL_C_INCLUDE_DIR ${SINGLE_LIB})
      ##   break()
      ## endforeach()
      ##set(HDF5_DIR ${HDF5_C_INCLUDE_DIR}/..)
      # vcpkg on MS Windows adds shlwapi.lib dependency to hdf5.lib
      set(HDF5_LIBRARIES ${HDF5_LIBRARIES} "shlwapi")
      set(HDF5_C_LIBRARIES ${HDF5_LIBRARIES} "shlwapi")
      set(HDF5_HL_LIBRARIES ${HDF5_HL_LIBRARIES} "shlwapi")
    endif(HDF5_FOUND)
  endif(USE_PKGCONFIG)
    
  if(HDF5_FOUND)
    message(STATUS "Found HDF5 by ${FOUND_BY}: ${HDF5_INCLUDE_DIRS}")
    include_directories(${HDF5_INCLUDE_DIRS})
    link_directories(${HDF5_LIBRARY_DIRS})
    link_directories(${HDF5_C_LIBRARY_DIRS})
    link_directories(${HDF5_HL_LIBRARY_DIRS})
    # HDF5 variable used in configure_file( ... gidpost_config.h.in)
    # set (HDF5 1)
  else(HDF5_FOUND)
    message(WARNING "*** HDF5 NOT FOUND. Please install HDF5 or define the HDF5DIR variable or make it available through pkg-config and define environment variable PKG_CONFIG_PATH.")
    message(WARNING "    disabling HDF5")
    set (ENABLE_HDF5 OFF)
  endif(HDF5_FOUND)
endif(ENABLE_HDF5)

if( NOT DEFINED ENABLE_SHARED_LIBS)
    option(ENABLE_SHARED_LIBS "Build gidpost as a shared libraries." OFF)
endif( NOT DEFINED ENABLE_SHARED_LIBS)

if (ENABLE_SHARED_LIBS)
  add_definitions (-DGIDPOST_SHARED)
endif (ENABLE_SHARED_LIBS)

# examples are not built by Kratos (see ../README_UPDATE.md); the Fortran
# binding and its compiler-detection block were dropped when re-vendoring
# gidpost 2.14 -- nothing in this tree references it (verified: absent from
# ../source/CMakeLists.txt's gidpost_source list).
if( NOT DEFINED ENABLE_EXAMPLES)
    option (ENABLE_EXAMPLES "Build examples, it is needed a C++ compiler" ON )
endif( NOT DEFINED ENABLE_EXAMPLES)

# used in examples/MultiplePartition
if ( NOT DEFINED ENABLE_PARALLEL_EXAMPLE)
    option(ENABLE_PARALLEL_EXAMPLE "Build gidpost as a shared libraries." OFF)
endif( NOT DEFINED ENABLE_PARALLEL_EXAMPLE)

if( ENABLE_PARALLEL_EXAMPLE )
  find_package( Boost COMPONENTS thread REQUIRED)
  if ( ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  endif ( ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
endif( ENABLE_PARALLEL_EXAMPLE )

# using g++ 11 in older linuxes
# to enable debug information
if (CMAKE_COMPILER_IS_GNUCXX)
  # get kernel version
  execute_process( COMMAND uname -r OUTPUT_VARIABLE UNAME_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
  message( -- " Kernel version: " ${UNAME_RESULT})
  string( REGEX MATCH "[0-9]+.[0-9]+" LINUX_KERNEL_VERSION ${UNAME_RESULT})
  # if ( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
  if ( LINUX_KERNEL_VERSION VERSION_LESS 5.0)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gdwarf-4")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")
    message(STATUS "NEW FLAGS = ${CMAKE_CXX_FLAGS}")
  # endif ( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11.0)
  endif ( LINUX_KERNEL_VERSION VERSION_LESS 5.0)
endif (CMAKE_COMPILER_IS_GNUCXX)

# configure a header file to pass some of the CMake settings
# to the source code. HAVE_GIDPOST_CONFIG_H and the include path to the
# generated header are target-scoped on the gidpost target itself (see
# source/CMakeLists.txt), not directory-scoped here, so the definitions
# gidpost needs internally never leak into consumers -- only the namespaced,
# PUBLIC KRATOS_GIDPOST_HAS_HDF5 macro does.
configure_file (
  "${PROJECT_SOURCE_DIR}/gidpost_config.h.in"
  "${PROJECT_BINARY_DIR}/gidpost_config.h"
)

install( FILES "${PROJECT_BINARY_DIR}/gidpost_config.h" DESTINATION "include" )

include_directories ("${PROJECT_SOURCE_DIR}")
include_directories ("${PROJECT_SOURCE_DIR}/source")

add_subdirectory (source)

# Make sure the linker can find the gidpost library once it is
# built.
link_directories (${PROJECT_BINARY_DIR}/source  ${ZLIB_LIBRARY_DIRS} ${HDF5_C_LIBRARY_DIRS} ${HDF5_HL_LIBRARY_DIRS})

# if (ENABLE_EXAMPLES)
#   add_subdirectory( examples )
#   add_subdirectory( examples/MultiplePartition )
# endif (ENABLE_EXAMPLES)
