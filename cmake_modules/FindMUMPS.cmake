# FindMUMPS.cmake
# Looks for dmumps_c.h and the MUMPS double-precision library set.
# Sets: MUMPS_FOUND, MUMPS_INCLUDE_DIRS, MUMPS_LIBRARIES, MUMPS_VERSION

# Common multiarch suffix paths (Debian/Ubuntu multiarch layout)
set(_MUMPS_MULTIARCH_SUFFIXES
    lib lib64
    lib/x86_64-linux-gnu lib/aarch64-linux-gnu
    lib/arm-linux-gnueabihf)

find_path(MUMPS_INCLUDE_DIR dmumps_c.h
    HINTS ${MUMPS_ROOT} ENV MUMPS_ROOT
    PATH_SUFFIXES include)

find_library(MUMPS_DOUBLE_LIB NAMES dmumps
    HINTS ${MUMPS_ROOT} ENV MUMPS_ROOT
    PATH_SUFFIXES ${_MUMPS_MULTIARCH_SUFFIXES})

find_library(MUMPS_COMMON_LIB NAMES mumps_common
    HINTS ${MUMPS_ROOT} ENV MUMPS_ROOT
    PATH_SUFFIXES ${_MUMPS_MULTIARCH_SUFFIXES})

find_library(MUMPS_PORD_LIB NAMES pord
    HINTS ${MUMPS_ROOT} ENV MUMPS_ROOT
    PATH_SUFFIXES ${_MUMPS_MULTIARCH_SUFFIXES})

# Sequential stub (no-MPI build): libmpiseq
find_library(MUMPS_SEQ_LIB NAMES mpiseq
    HINTS ${MUMPS_ROOT} ENV MUMPS_ROOT
    PATH_SUFFIXES ${_MUMPS_MULTIARCH_SUFFIXES} libseq)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUMPS
    REQUIRED_VARS MUMPS_INCLUDE_DIR MUMPS_DOUBLE_LIB MUMPS_COMMON_LIB)

if(MUMPS_FOUND)
    set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR})
    set(MUMPS_LIBRARIES
        ${MUMPS_DOUBLE_LIB}
        ${MUMPS_COMMON_LIB}
        ${MUMPS_PORD_LIB})
    if(MUMPS_SEQ_LIB)
        list(APPEND MUMPS_LIBRARIES ${MUMPS_SEQ_LIB})
    endif()
endif()

mark_as_advanced(MUMPS_INCLUDE_DIR MUMPS_DOUBLE_LIB MUMPS_COMMON_LIB
                 MUMPS_PORD_LIB MUMPS_SEQ_LIB)
