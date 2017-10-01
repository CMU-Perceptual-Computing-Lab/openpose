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
