set(SOURCES_OP_3D
    cameraParameters.cpp
    pointGrey.cpp
    reconstruction3D.cpp
    renderer.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_3D_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_3D})
set(SOURCES_OP_3D_WITH_CP ${SOURCES_OP_3D_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_3D_WITH_CP} PARENT_SCOPE)

if (UNIX AND NOT APPLE)
	add_library(openpose_3d ${SOURCES_OP_3D})

	set_target_properties(openpose_3d PROPERTIES COMPILE_FLAGS ${OP_CXX_FLAGS})

	install(TARGETS openpose_3d
	    EXPORT OpenPose
	    RUNTIME DESTINATION bin
	    LIBRARY DESTINATION lib
	    ARCHIVE DESTINATION lib/openpose)
endif (UNIX AND NOT APPLE)
