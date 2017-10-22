# add_subdirectory(core)
# add_subdirectory(face)
# add_subdirectory(filestream)
# add_subdirectory(gui)
# add_subdirectory(hand)
# add_subdirectory(pose)
# add_subdirectory(producer)
# add_subdirectory(thread)
# add_subdirectory(utilities)
# add_subdirectory(wrapper)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
subdirlist(SUB_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
foreach (SUB_DIR ${SUB_DIRS})
  add_subdirectory(${SUB_DIR})
endforeach (SUB_DIR ${SUB_DIRS})

# message(STATUS "here \${SOURCES_OPENPOSE} = ${SOURCES_OPENPOSE}")

if (CMAKE_COMPILER_IS_GNUCXX)
set_target_properties(openpose_core openpose_filestream openpose_gui openpose_pose 
    openpose_utilities openpose_producer openpose_thread openpose_face openpose_hand 
    openpose_wrapper
  PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})
endif (CMAKE_COMPILER_IS_GNUCXX)

# make the openpose.so

# TODO -- These comments will be removed soon.
# file(GLOB_RECURSE SOURCES "*.cu" "*.cpp") # It's better not to hardcode here.
# message(STATUS "\${SOURCES} ${SOURCES}")

cuda_add_library(openpose ${SOURCES_OPENPOSE})
message("\${SOURCES_OP_GUI_WITH_CP} = ${SOURCES_OP_GUI_WITH_CP}")

if (WIN32)
  foreach (SUB_DIR ${SUB_DIRS})
    string(TOUPPER ${SUB_DIR} SUB_DIR_UPPERCASE)
    source_group(${SUB_DIR} FILES ${SOURCES_OP_${SUB_DIR_UPPERCASE}_WITH_CP})
  endforeach (SUB_DIR ${SUB_DIRS})
endif (WIN32)

target_link_libraries(openpose ${OpenCV_LIBS} debug ${Boost_FILESYSTEM_LIBRARY_DEBUG} debug ${Boost_SYSTEM_LIBRARY_DEBUG}
		optimized ${Boost_FILESYSTEM_LIBRARY_OPTIMIZED} optimized ${Boost_SYSTEM_LIBRARY_OPTIMIZED} ${Caffe_LIBS}
		debug ${GFLAGS_LIBRARY_DEBUG} optimized ${GFLAGS_LIBRARY_RELEASE}
		debug ${GLOG_LIBRARY_DEBUG} optimized ${GLOG_LIBRARY_RELEASE})

if (WIN32)
  set_target_properties(openpose PROPERTIES COMPILE_FLAGS -DOP_EXPORTS)
endif (WIN32)

if (UNIX AND NOT APPLE)
 
  add_library(caffe SHARED IMPORTED)
	set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS})
	target_link_libraries(openpose caffe)

  if (BUILD_CAFFE)
  	add_dependencies(openpose openpose_caffe)
  endif (BUILD_CAFFE)
endif (UNIX AND NOT APPLE)

set_property(TARGET openpose PROPERTY VERSION ${OpenPose_VERSION})

install(TARGETS openpose
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
