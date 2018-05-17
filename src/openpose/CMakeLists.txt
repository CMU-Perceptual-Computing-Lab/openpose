include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
subdirlist(SUB_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
foreach (SUB_DIR ${SUB_DIRS})
  add_subdirectory(${SUB_DIR})
endforeach (SUB_DIR ${SUB_DIRS})

# make the openpose.so/dll
file(GLOB_RECURSE OP_HEADERS_UNFILTERED "${CMAKE_SOURCE_DIR}/include/openpose/*.h"
    "${CMAKE_SOURCE_DIR}/include/openpose/*.hpp"
    "${CMAKE_SOURCE_DIR}/include/openpose/*.hu")

set(OP_HEADERS "")
# # Add experimental sub-folders
# foreach (FILE ${OP_HEADERS_UNFILTERED})
#   if (NOT FILE MATCHES "${CMAKE_SOURCE_DIR}/include/openpose/experimental/*")
#       list(APPEND OP_HEADERS ${FILE})
#   endif ()
# endforeach(FILE ${OP_HEADERS_UNFILTERED})

if (${GPU_MODE} MATCHES "CUDA")
    cuda_add_library(openpose ${SOURCES_OPENPOSE} ${OP_HEADERS})
else()
    add_library(openpose ${SOURCES_OPENPOSE} ${OP_HEADERS})
endif ()

# Ubuntu
if (UNIX AND NOT APPLE)
  target_link_libraries(openpose ${OpenPose_3rdparty_libraries})
  if (CMAKE_COMPILER_IS_GNUCXX)
    foreach (SUB_DIR ${SUB_DIRS})
        set_target_properties(openpose_${SUB_DIR} PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})
    endforeach (SUB_DIR ${SUB_DIRS})
    set_target_properties(openpose PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})
  endif (CMAKE_COMPILER_IS_GNUCXX)
# Windows
elseif (WIN32)
  set_property(TARGET openpose PROPERTY DEBUG_POSTFIX d)
  target_link_libraries(openpose ${OpenPose_3rdparty_libraries})
  if (${DL_FRAMEWORK} MATCHES "CAFFE")
    target_compile_definitions(openpose PRIVATE BOOST_ALL_NO_LIB)
  endif(${DL_FRAMEWORK} MATCHES "CAFFE")
  set_property(TARGET openpose PROPERTY FOLDER "OpenPose library")

  foreach (SUB_DIR ${SUB_DIRS})
    string(TOUPPER ${SUB_DIR} SUB_DIR_UPPERCASE)
    source_group("Source Files\\${SUB_DIR}" FILES ${SOURCES_OP_${SUB_DIR_UPPERCASE}_WITH_CP} "*")
    file(GLOB_RECURSE SUB_MOD_HEADERS "${CMAKE_SOURCE_DIR}/include/openpose/${SUB_DIR}/*.h"
        "${CMAKE_SOURCE_DIR}/include/openpose/${SUB_DIR}/*.hpp"
        "${CMAKE_SOURCE_DIR}/include/openpose/${SUB_DIR}/*.hu")
    source_group("Header Files\\${SUB_DIR}" FILES ${SUB_MOD_HEADERS})
  endforeach (SUB_DIR ${SUB_DIRS})
  set_target_properties(openpose PROPERTIES COMPILE_FLAGS -DOP_EXPORTS)
  # Properties->General->Output Directory
  set_property(TARGET openpose PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
  set_property(TARGET openpose PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))

endif (UNIX AND NOT APPLE)

if (UNIX AND NOT APPLE)
  add_library(caffe SHARED IMPORTED)
  set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS})
  target_link_libraries(openpose caffe)

  if (BUILD_CAFFE)
    add_dependencies(openpose openpose_caffe)
  endif (BUILD_CAFFE)

  set_property(TARGET openpose PROPERTY VERSION ${OpenPose_VERSION})

  install(TARGETS openpose
      EXPORT OpenPose
      RUNTIME DESTINATION bin
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
