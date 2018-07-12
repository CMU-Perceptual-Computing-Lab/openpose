set(SOURCES_OP_THREAD
    defineTemplates.cpp)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
prepend(SOURCES_OP_THREAD_WITH_CP ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCES_OP_THREAD})
set(SOURCES_OP_THREAD_WITH_CP ${SOURCES_OP_THREAD_WITH_CP} PARENT_SCOPE)
set(SOURCES_OPENPOSE ${SOURCES_OPENPOSE} ${SOURCES_OP_THREAD_WITH_CP} PARENT_SCOPE)

if (UNIX OR APPLE)
	add_library(openpose_thread ${SOURCES_OP_THREAD})

    target_link_libraries(openpose_thread openpose_core)
	
	install(TARGETS openpose_thread 
	    EXPORT OpenPose
	    RUNTIME DESTINATION bin
	    LIBRARY DESTINATION lib
	    ARCHIVE DESTINATION lib/openpose)
endif (UNIX OR APPLE)
