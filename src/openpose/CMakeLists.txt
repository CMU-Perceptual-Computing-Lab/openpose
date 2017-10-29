include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
subdirlist(SUB_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
foreach (SUB_DIR ${SUB_DIRS})
  add_subdirectory(${SUB_DIR})
endforeach (SUB_DIR ${SUB_DIRS})

<<<<<<< HEAD
# TODO -- Test this on Ubuntu.
if (CMAKE_COMPILER_IS_GNUCXX)
  foreach (SUB_DIR ${SUB_DIRS})
      set_target_properties(openpose_${SUB_DIR} PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})
  endforeach (SUB_DIR ${SUB_DIRS})
  set_target_properties(openpose PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})
endif (CMAKE_COMPILER_IS_GNUCXX)

=======
>>>>>>> 5a640b34e58050acb0eff2dcb2024a8b17185db6
# make the openpose.so/dll
file(GLOB_RECURSE OP_HEADERS_UNFILTERED "${CMAKE_SOURCE_DIR}/include/openpose/*.h" 
        "${CMAKE_SOURCE_DIR}/include/openpose/*.hpp" 
        "${CMAKE_SOURCE_DIR}/include/openpose/*.hu")
set(OP_HEADERS "")
foreach (FILE ${OP_HEADERS_UNFILTERED})
  if (NOT FILE MATCHES "${CMAKE_SOURCE_DIR}/include/openpose/experimental/*")
      list(APPEND OP_HEADERS ${FILE})
  endif ()
endforeach(FILE ${OP_HEADERS_UNFILTERED})
# list(REMOVE_ITEM OP_HEADERS "${CMAKE_SOURCE_DIR}/include/openpose/experimental/*")

<<<<<<< HEAD
message(STATUS "\${OP_HEADERS} = ${OP_HEADERS}")
message(STATUS "\${OP_HEADERS_UNFILTERED} = ${OP_HEADERS_UNFILTERED}")

cuda_add_library(openpose ${SOURCES_OPENPOSE} ${OP_HEADERS})
target_link_libraries(openpose ${OpenCV_LIBS} ${Caffe_LIBS}
    debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} debug ${Boost_SYSTEM_LIBRARY_DEBUG}
    optimized ${Boost_FILESYSTEM_LIBRARY_OPTIMIZED} optimized ${Boost_SYSTEM_LIBRARY_OPTIMIZED} 
    debug ${GFLAGS_LIBRARY_DEBUG} optimized ${GFLAGS_LIBRARY_RELEASE}
    debug ${GLOG_LIBRARY_DEBUG} optimized ${GLOG_LIBRARY_RELEASE})

if (WIN32)
=======
cuda_add_library(openpose ${SOURCES_OPENPOSE} ${OP_HEADERS})

if(UNIX AND NOT APPLE)
  target_link_libraries(openpose ${OpenCV_LIBS} ${Caffe_LIBS}
      ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} 
      ${GFLAGS_LIBRARY} ${GLOG_LIBRARY})
  if (CMAKE_COMPILER_IS_GNUCXX)
    foreach (SUB_DIR ${SUB_DIRS})
        set_target_properties(openpose_${SUB_DIR} PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})
    endforeach (SUB_DIR ${SUB_DIRS})
    set_target_properties(openpose PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})
  endif (CMAKE_COMPILER_IS_GNUCXX)
endif(UNIX AND NOT APPLE)

if (WIN32)
  target_link_libraries(openpose ${OpenCV_LIBS} ${Caffe_LIBS}
      debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} debug ${Boost_SYSTEM_LIBRARY_DEBUG}
      optimized ${Boost_FILESYSTEM_LIBRARY_OPTIMIZED} optimized ${Boost_SYSTEM_LIBRARY_OPTIMIZED} 
      debug ${GFLAGS_LIBRARY_DEBUG} optimized ${GFLAGS_LIBRARY_RELEASE}
      debug ${GLOG_LIBRARY_DEBUG} optimized ${GLOG_LIBRARY_RELEASE})
>>>>>>> 5a640b34e58050acb0eff2dcb2024a8b17185db6
  target_compile_definitions(openpose PRIVATE BOOST_ALL_NO_LIB)
  # source_group("Header Files" FILES ${CMAKE_SOURCE_DIR}/include)
  set_property(TARGET openpose PROPERTY FOLDER "Openpose library")
  foreach (SUB_DIR ${SUB_DIRS})
    string(TOUPPER ${SUB_DIR} SUB_DIR_UPPERCASE)
    source_group("Source Files\\${SUB_DIR}" FILES ${SOURCES_OP_${SUB_DIR_UPPERCASE}_WITH_CP} "*")
    message(${CMAKE_SOURCE_DIR}/include/openpose/${SUB_DIR})
    file(GLOB_RECURSE SUB_MOD_HEADERS "${CMAKE_SOURCE_DIR}/include/openpose/${SUB_DIR}/*.h" 
        "${CMAKE_SOURCE_DIR}/include/openpose/${SUB_DIR}/*.hpp" 
        "${CMAKE_SOURCE_DIR}/include/openpose/${SUB_DIR}/*.hu")
    message(STATUS "\${SUB_MOD_HEADERS} = ${SUB_MOD_HEADERS}")
    source_group("Header Files\\${SUB_DIR}" FILES ${SUB_MOD_HEADERS})
  endforeach (SUB_DIR ${SUB_DIRS})
  set_target_properties(openpose PROPERTIES COMPILE_FLAGS -DOP_EXPORTS)
endif (WIN32)

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
<<<<<<< HEAD
endif (UNIX AND NOT APPLE)
=======
endif (UNIX AND NOT APPLE)
>>>>>>> 5a640b34e58050acb0eff2dcb2024a8b17185db6
