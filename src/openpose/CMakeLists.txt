add_subdirectory(core)
add_subdirectory(filestream)
add_subdirectory(gui)
add_subdirectory(pose)
add_subdirectory(utilities)
add_subdirectory(producer)
add_subdirectory(thread)
add_subdirectory(face)
add_subdirectory(hand)
add_subdirectory(wrapper)
set_target_properties(openpose_core openpose_filestream openpose_gui openpose_pose 
    openpose_utilities openpose_producer openpose_thread openpose_face openpose_hand 
    openpose_wrapper
  PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})

# make the openpose.so

add_library(caffe SHARED IMPORTED)
set_property(TARGET caffe PROPERTY IMPORTED_LOCATION ${Caffe_LIBS})

file(GLOB_RECURSE SOURCES "*.cu" "*.cpp") # It's better not to hardcode here.

cuda_add_library(openpose ${SOURCES})
target_link_libraries(openpose ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} caffe)
if (BUILD_CAFFE)
    add_dependencies(openpose openpose_caffe)
endif (BUILD_CAFFE)

set_property(TARGET openpose PROPERTY VERSION ${OpenPose_VERSION})

install(TARGETS openpose
    EXPORT OpenPose
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/openpose)
