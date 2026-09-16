#include (ExternalProject)

message( STATUS "* Configuring gidpost library" )

set (gidpost_source 
  gidpost.c gidpost_cluster.c gidpostFILES.c gidpostInt.c
  gidpostHash.c hashtab.c recycle.c lookupa.c)

file(GLOB SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

foreach(file ${SOURCE_FILES})
    set_source_files_properties(${file} PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE)
endforeach()

# find  package already done in  ../CmakeLists.txt
# find_package(HDF5)
# if (HDF5_FOUND)
#   option (ADD_HDF5_SUPPORT "Use HDF5" ON)
# endif(HDF5_FOUND)

if (ENABLE_HDF5)
  set (hdf5_source hdf5c.c gidpostHDF5.c)
endif (ENABLE_HDF5)

if (ENABLE_SHARED_LIBS)
  set (LIB_TYPE SHARED)
else (ENABLE_SHARED_LIBS)
  set (LIB_TYPE STATIC)
endif (ENABLE_SHARED_LIBS)

add_library (gidpost ${LIB_TYPE} ${gidpost_source} ${hdf5_source})

# Internal to gidpost's own compilation: tells gidpost.h to pull in the
# generated gidpost_config.h (which only carries #cmakedefine ENABLE_HDF5,
# see ../gidpost_config.h.in). PRIVATE on purpose -- this is a generic,
# un-namespaced macro name and must not leak into every consumer translation
# unit (KratosCore links this target PUBLIC).
target_compile_definitions(gidpost PRIVATE HAVE_GIDPOST_CONFIG_H)
target_include_directories(gidpost PRIVATE ${PROJECT_BINARY_DIR})

if (ENABLE_SHARED_LIBS)
  add_definitions (-DGIDPOST_EXPORTS)
endif (ENABLE_SHARED_LIBS)

if (MSVC)
  add_definitions (-D_CRT_SECURE_NO_DEPRECATE)
endif (MSVC)

# if()
#   set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")
# endif()
if ( ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR ( CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  target_compile_options( gidpost PRIVATE "$<$<CONFIG:Debug>:-Wall>")
  target_compile_options( gidpost PRIVATE "$<$<CONFIG:Debug>:-Wextra>")
  target_compile_options( gidpost PRIVATE "$<$<CONFIG:Debug>:-Wconversion>")
  target_compile_options( gidpost PRIVATE "$<$<CONFIG:Debug>:-Wno-unused-parameter>")
  # add_compile_options( "$<$<CONFIG:Debug>:-Wall>")
  # add_compile_options( "$<$<CONFIG:Debug>:-Wextra>")
endif()

if (ENABLE_HDF5)
  # gidpost's own sources see ENABLE_HDF5 through the generated gidpost_config.h
  # (HAVE_GIDPOST_CONFIG_H, set above) -- not redefined here, which would just
  # warn on every translation unit. KRATOS_GIDPOST_HAS_HDF5 is the namespaced,
  # PUBLIC macro that actually needs to reach consumers -- it is what lets
  # kratos/includes/gid_io.h detect at compile time whether GiD_PostHDF5 is
  # usable, which is otherwise impossible: gidpost_config.h is not on a
  # consumer's include path.
  target_compile_definitions(gidpost PUBLIC KRATOS_GIDPOST_HAS_HDF5)
  target_include_directories(gidpost PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(gidpost PRIVATE ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif (ENABLE_HDF5)

# find  package already done in  ../CmakeLists.txt
# find_package(ZLIB REQUIRED)
if (ZLIB_FOUND)
  target_include_directories(gidpost PRIVATE ${ZLIB_INCLUDE_DIRS})
  target_link_libraries(gidpost PRIVATE ${ZLIB_LIBRARIES})
else ( ZLIB_FOUND )
  message ( SEND_ERROR "ZLIB_LIBRARY NOTFOUND, set ZLIB_INCLUDE_DIRS and ZLIB_LIBRARIES variables")
endif(ZLIB_FOUND)

find_package( Threads )
if ( Threads_FOUND )
  target_link_libraries(gidpost PRIVATE ${CMAKE_THREAD_LIBS_INIT})
endif (Threads_FOUND)

if (UNIX)
  target_link_libraries(gidpost PRIVATE m)
endif (UNIX)

# install rules
#
install (TARGETS gidpost
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install( FILES gidpost.h gidpost_types.h gidpost_functions.h gidpost_cluster_functions.h gidpost_functions_deprecated.h DESTINATION include )
